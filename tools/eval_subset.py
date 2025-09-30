#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from pycocotools.coco import COCO

from damo.base_models.core.ops import RepConv
from damo.apis.detector_inference import inference
from damo.config.base import parse_config
from damo.dataset import build_dataloader, build_dataset
from damo.detectors.detector import build_local_model
from damo.utils import fuse_model, setup_logger

from my_help_functions.hooks import remove_all_hooks
import torch.nn.functional as F


def mkdir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path)


def make_parser():
    parser = argparse.ArgumentParser('damo eval (subset)')

    # distributed
    parser.add_argument('--local_rank', type=int, default=0)

    # core params (оставляем как в tools/eval.py)
    parser.add_argument('-f', '--config_file', default=None, type=str,
                        help='config file path')
    parser.add_argument('-c', '--ckpt', default=None, type=str,
                        help='ckpt for eval')
    parser.add_argument('--conf', default=None, type=float, help='test conf')
    parser.add_argument('--nms', default=None, type=float, help='nms thr')
    parser.add_argument('--tsize', default=None, type=int, help='test img size')
    parser.add_argument('--seed', default=None, type=int, help='eval seed')
    parser.add_argument('--fuse', dest='fuse', action='store_true',
                        help='fuse conv+bn before eval')
    parser.add_argument('--batch_size', type=int, default=8)

    # subset controls
    parser.add_argument('--subset_size', type=int, default=128,
                        help='Сколько изображений брать (первые N по алфавиту)')

    # passthrough options override
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help='Modify config options via CLI')
    return parser


class DatasetView(torch.utils.data.Dataset):
    """Прозрачный прокси над датасетом с переупорядоченным и урезанным индексом.
    Ключевые моменты:
    - __getitem__/pull_item/get_img_info/load_anno вызывают базовый датасет напрямую.
    - __setattr__ прокидывает присваивания (например, _transforms) в базовый датасет.
    - Подрезаем также объект COCO (`.coco.dataset`) — тогда COCOeval будет считать метрики
      только по подмножеству изображений.
    - В __getstate__/__setstate__ сохраняем только словарь датасета COCO, чтобы корректно
      сериализовать объект через multiprocessing (Windows spawn).
    """
    def __init__(self, base_ds, indices: List[int]):
        object.__setattr__(self, "_base", base_ds)
        object.__setattr__(self, "_indices", list(indices))

        # Собираем подмножество image ids (в COCO обычно хранится в .ids или .img_ids)
        base_ids = None
        if hasattr(base_ds, 'ids'):
            base_ids = list(getattr(base_ds, 'ids'))
        elif hasattr(base_ds, 'img_ids'):
            base_ids = list(getattr(base_ds, 'img_ids'))
        if base_ids is not None:
            subset_ids = [base_ids[i] for i in indices]
        else:
            subset_ids = None
        object.__setattr__(self, "_ids", subset_ids)

        # Если базовый датасет имеет pycocotools COCO, создаём его урезанную копию
        if subset_ids is not None and hasattr(base_ds, 'coco') and getattr(base_ds, 'coco') is not None:
            orig_coco = base_ds.coco
            imgs = [img for img in orig_coco.dataset.get('images', []) if img.get('id') in subset_ids]
            anns = [ann for ann in orig_coco.dataset.get('annotations', []) if ann.get('image_id') in subset_ids]
            cats = orig_coco.dataset.get('categories', [])
            new_dataset = {'images': imgs, 'annotations': anns, 'categories': cats}
            # сохраняем необязательные поля
            for k in ('info', 'licenses', 'type'):
                if k in orig_coco.dataset:
                    new_dataset[k] = orig_coco.dataset[k]
            new_coco = COCO()
            new_coco.dataset = new_dataset
            new_coco.createIndex()
            object.__setattr__(self, "_coco", new_coco)
            logger.info(f'COCO subset created: images={len(imgs)} annotations={len(anns)} categories={len(cats)}')
        else:
            object.__setattr__(self, "_coco", None)

    # ---- публичный интерфейс ----
    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        # Дёргаем именно метод базы, чтобы сохранить все трансформации/типы
        return self._base.__getitem__(self._indices[idx])

    def get_img_info(self, idx):
        return self._base.get_img_info(self._indices[idx])

    def load_anno(self, idx):
        if hasattr(self._base, 'load_anno'):
            return self._base.load_anno(self._indices[idx])
        raise AttributeError('Underlying dataset has no load_anno')

    def pull_item(self, idx):
        if hasattr(self._base, 'pull_item'):
            return self._base.pull_item(self._indices[idx])
        raise AttributeError('Underlying dataset has no pull_item')

    # ---- то, что ожидает COCOeval ----
    @property
    def ids(self):  # наиболее распространённое имя
        return self._ids if self._ids is not None else getattr(self._base, 'ids')

    @property
    def img_ids(self):  # на случай альтернативного имени
        if self._ids is not None:
            return self._ids
        return getattr(self._base, 'img_ids') if hasattr(self._base, 'img_ids') else getattr(self._base, 'ids')

    @property
    def coco(self):
        return self._coco if self._coco is not None else getattr(self._base, 'coco')

    # ---- делегирование/сериализация ----
    def __getattr__(self, name):
        if name in {"_base", "_indices", "_ids", "_coco"}:
            raise AttributeError
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        if name in {"_base", "_indices", "_ids", "_coco"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._base, name, value)

    def __getstate__(self):
        # сериализуем базу и только словарь COCO (не сам объект COCO)
        state = {"_base": self._base, "_indices": self._indices, "_ids": self._ids}
        state["_coco_dataset"] = self._coco.dataset if getattr(self, '_coco', None) is not None else None
        return state

    def __setstate__(self, state):
        object.__setattr__(self, "_base", state["_base"])
        object.__setattr__(self, "_indices", state["_indices"])
        object.__setattr__(self, "_ids", state.get("_ids", None))
        coco_ds = state.get("_coco_dataset", None)
        if coco_ds is not None:
            new_coco = COCO()
            new_coco.dataset = coco_ds
            new_coco.createIndex()
            object.__setattr__(self, "_coco", new_coco)
        else:
            object.__setattr__(self, "_coco", None)


def build_alphabetical_subset(ds, n: int):
    total = len(ds)
    if n is None or n <= 0 or n >= total:
        logger.info(f'\nSubset disabled: using all {total} images.')
        return ds

    # Собираем пары (index, file_name) и сортируем по имени
    names = []
    for i in range(total):
        info = ds.get_img_info(i)
        fname = info.get('file_name', '')
        names.append((i, fname))

    names.sort(key=lambda t: t[1])
    selected = [i for i, _ in names[:n]]

    # Логи для контроля
    sample_preview = [names[k][1] for k in range(min(5, len(selected)))]
    logger.info(
        f'\nAlphabetical subset: picked {len(selected)} of {total} images.'
        f'\nFirst few file_names: {sample_preview}'
    )
    return DatasetView(ds, selected)


def hook_bn_modified(name):
    """
    Модифицированный хук для BatchNorm слоя - только применяет scale и shift
    """
    def hook_fn(module, input, output):
        
        modified_output = input[0] * module.weight.view(1, -1, 1, 1) + module.bias.view(1, -1, 1, 1)
                
        return modified_output
    
    return hook_fn

def precompute_pre_center(model):
    modules = list(model.named_modules())
    for i in range(len(modules) - 1):
        name1, conv = modules[i]
        name2, bn = modules[i + 1]

        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            with torch.no_grad():

                # W = conv.weight       # [out_ch, in_ch, kH, kW]
                # out_ch, in_ch, kH, kW = W.shape
                # center = bn.running_mean
                # W_eff = W.view(W.size()[0], W.size()[1], -1).sum(dim=2)  # shape: [out_ch, in_ch]
                # W_pinv = torch.pinverse(W_eff)        # [in_ch, out_ch]
                # center_before = W_pinv @ center
                # conv.register_buffer("_pre_center", center_before)

                # W = conv.weight       # [out_ch, in_ch, kH, kW]
                # out_ch, in_ch, kH, kW = W.shape
                # center = bn.running_mean
                # W_fl = W.flatten(1)
                # W_pinv = torch.pinverse(W_fl)        # [k * k * in_ch, out_ch]
                # center_before = W_pinv @ center
                # center_before = center_before.view(in_ch, kH * kW).mean(1)
                # conv.register_buffer("_pre_center", center_before)

                W = conv.weight       # [out_ch, in_ch, kH, kW]
                out_ch, in_ch, kH, kW = W.shape
                center = bn.running_mean
                W_fl = W.flatten(1)
                U, S, Vh = torch.svd(W_fl)
                U_inv = torch.pinverse(U)
                print(Vh.size(), out_ch, in_ch, kH, kW)
                Vh_inv = torch.pinverse(Vh)
                print(Vh_inv.size(), U_inv.size())
                Vh_inv = Vh_inv.view(Vh_inv.size()[0], in_ch, kH, kW).mean(dim=(2, 3)).permute(1, 0)
                center_before_U = U_inv @ center
                center_before_S = center_before_U / S
                center_before_Vh = Vh_inv @ center_before_S

                conv.register_buffer("_pre_center", center_before_Vh)



def hook_conv_modified_pre(name, bn_layer):
    def hook_fn(module, input, output):
        with torch.no_grad():
            x = input[0]
 
            y = x - module._pre_center.view(1, -1, 1, 1)

            y = torch.conv2d(
                y, 
                module.weight, 
                module.bias, 
                module.stride, 
                module.padding, 
                module.dilation, 
                module.groups
            )

            if module.bias is not None:
                y = y + module.bias.view(1, -1, 1, 1)

            # BN scale
            # y = y - bn_layer.running_mean.view(1, -1, 1, 1)
            y = y / torch.sqrt(bn_layer.running_var.view(1, -1, 1, 1) + bn_layer.eps)
            return y

    return hook_fn

def register_conv_bn_hooks_modified(model):
    """
    Регистрирует модифицированные хуки для всех пар conv-bn
    """
    
    remove_all_hooks(model)
    modules = list(model.named_modules())

    precompute_pre_center(model)

    for i in range(len(modules) - 1):
        name1, layer1 = modules[i]
        name2, layer2 = modules[i + 1]
        
        if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.BatchNorm2d):
            layer1.register_forward_hook(hook_conv_modified_pre(name1, layer2))
            layer2.register_forward_hook(hook_bn_modified(name2))


'''
def svd_forward_hook_conv(name, keep_ratio, bn_layer):
    """
    Хук для Conv2d: аппроксимация весов через усечённое SVD.
    keep_ratio — доля сингулярных компонент, которые нужно оставить (0..1).
    """
    def hook_fn(module, input, output):
        with torch.no_grad():
            W = module.weight       # [out_ch, in_ch, kH, kW]
            out_ch, in_ch, kH, kW = W.shape
            W_flat = W.view(out_ch, -1) # [out_ch, in_ch * kH * kW]
            U, S, Vh = torch.svd(W_flat)
            rank = max(1, int(len(S) * keep_ratio))

            center = bn_layer.running_mean

            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:, :rank]
            W_approx = (U_r @ torch.diag(S_r) @ Vh_r.T).view_as(W)

            x = input[0]

            # W_eff = W_approx.view(W_approx.size()[0], W_approx.size()[1], -1).sum(dim=2)  # shape: [out_ch, in_ch]
            # W_pinv = torch.pinverse(W_eff)
            # center_before_conv = W_pinv @ center

            # signal_after_center = x - center_before_conv.view(1, -1, 1, 1)

            # x = F.conv2d(
            #     signal_after_center, W_approx, module.bias,
            #     stride=module.stride,
            #     padding=module.padding,
            #     dilation=module.dilation,
            #     groups=module.groups
            # )

            # modified_output = x / torch.sqrt(bn_layer.running_var.view(1, -1, 1, 1) + bn_layer.eps)
            # return modified_output

            dotp = torch.abs(U_r.T @ center)
            correlated   = torch.nonzero(dotp >= S_r[-1], as_tuple=True)[0]
            uncorrelated = torch.nonzero(dotp <  S_r[-1], as_tuple=True)[0]

            # =========== correlated =============

            U_corr = U_r[:, correlated]
            S_corr = S_r[correlated]
            Vh_corr = Vh_r[:, correlated]

            # W_correlated = (U_corr @ torch.diag(S_corr) @ Vh_corr.T).view_as(W_approx)

            # W_eff = W_correlated.view(W_correlated.size()[0], W_correlated.size()[1], -1).sum(dim=2)  # shape: [out_ch, in_ch]
            # W_pinv = torch.pinverse(W_eff)
            # center_before_conv_corr = W_pinv @ center

            # signal_corr_after_center = x - center_before_conv_corr.view(1, -1, 1, 1)

            # signal_after_conv_correlated = F.conv2d(
            #     signal_corr_after_center, W_correlated, module.bias,
            #     stride=module.stride,
            #     padding=module.padding,
            #     dilation=module.dilation,
            #     groups=module.groups
            # )
            

            
            W1_corr = Vh_corr.T.contiguous().view(len(correlated), in_ch // module.groups, kH, kW)
            signal_after_hidden_conv_corr = F.conv2d(x, W1_corr,
                                                      stride=module.stride,
                                                      padding=module.padding,
                                                      dilation=module.dilation,
                                                      groups=module.groups
                                                      )

            W2_corr = (U_corr * S_corr).contiguous().view(out_ch, len(correlated), 1, 1)
            # signal_after_conv_correlated = F.conv2d(signal_after_hidden_conv_corr, W2_corr,
            #                                  stride=1,
            #                                  padding=0,
            #                                  bias=module.bias
            #                                  )

            # ------------ center ----------------
            W_eff = W2_corr.view(W2_corr.size()[0], W2_corr.size()[1], -1).sum(dim=2)  # shape: [out_ch, in_ch]
            W_pinv = torch.pinverse(W_eff)
            center_before_conv_corr = W_pinv @ center

            signal_corr_after_center = signal_after_hidden_conv_corr - center_before_conv_corr.view(1, -1, 1, 1)
            signal_after_conv_correlated = F.conv2d(signal_corr_after_center, W2_corr,
                                             stride=1,
                                             padding=0,
                                             bias=module.bias
                                             )
            
            # ================== uncorrelated =================================
            U_uncorr = U_r[:, uncorrelated]
            S_uncorr = S_r[uncorrelated]
            Vh_uncorr = Vh_r[:, uncorrelated]

            # W_uncorrelated = (U_uncorr @ torch.diag(S_uncorr) @ Vh_uncorr.T).view_as(W_approx)

            # W_eff = W_uncorrelated.view(W_uncorrelated.size()[0], W_uncorrelated.size()[1], -1).sum(dim=2)  # shape: [out_ch, in_ch]
            # W_pinv = torch.pinverse(W_eff)
            # center_before_conv_uncorr = W_pinv @ center

            # signal_uncorr_after_center = x - center_before_conv_uncorr.view(1, -1, 1, 1)

            # signal_after_conv_uncorrelated = F.conv2d(
            #     signal_uncorr_after_center, W_uncorrelated, module.bias,
            #     stride=module.stride,
            #     padding=module.padding,
            #     dilation=module.dilation,
            #     groups=module.groups
            # )
            
            
            W1_uncorr = Vh_uncorr.T.contiguous().view(len(uncorrelated), in_ch // module.groups, kH, kW)
            signal_after_hidden_conv_uncorr = F.conv2d(x, W1_uncorr,
                                                      stride=module.stride,
                                                      padding=module.padding,
                                                      dilation=module.dilation,
                                                      groups=module.groups
                                                      )

            W2_uncorr = (U_uncorr * S_uncorr).contiguous().view(out_ch, len(uncorrelated), 1, 1)
            # signal_after_conv_uncorrelated = F.conv2d(signal_after_hidden_conv_uncorr, W2_uncorr,
            #                                  stride=1,
            #                                  padding=0,
            #                                  bias=module.bias
            #                                  )

            # ---------- center ----------
            W_eff = W2_uncorr.view(W2_uncorr.size()[0], W2_uncorr.size()[1], -1).sum(dim=2)  # shape: [out_ch, in_ch]
            W_pinv = torch.pinverse(W_eff)
            center_before_conv_uncorr = W_pinv @ center


            signal_uncorr_after_center = signal_after_hidden_conv_uncorr # - center_before_conv_uncorr.view(1, -1, 1, 1)
            signal_after_conv_uncorrelated = F.conv2d(signal_uncorr_after_center, W2_uncorr,
                                             stride=1,
                                             padding=0,
                                             bias=module.bias
                                             )
            
            # ============ final ================================================================
            output = signal_after_conv_correlated + signal_after_conv_uncorrelated 
            # modified_output = (output - bn_layer.running_mean.view(1, -1, 1, 1)) \
            # / torch.sqrt(bn_layer.running_var.view(1, -1, 1, 1) + bn_layer.eps)
            modified_output = output / torch.sqrt(bn_layer.running_var.view(1, -1, 1, 1) + bn_layer.eps)

            print('uncorr center -1: ', torch.norm(center_before_conv_uncorr))
            print('corr center -1: ', torch.norm(center_before_conv_corr))
            print('center', torch.norm(center))
            print(*list(map(torch.norm, (W1_corr, W2_corr, W1_uncorr, W2_uncorr))))

            return modified_output

    return hook_fn

def register_svd_hooks(model, layers, keep_ratio):
    remove_all_hooks(model)
    for layer_to_add in layers:
        for name, layer in model.named_modules():
            if layer_to_add in name:
                layer.register_forward_hook(lambda m, inp, out: svd_forward_hook(m, inp, out, keep_ratio=keep_ratio))
'''


def precompute_svd_weights(model, keep_ratio):
    global counter_1x1
    global counter_3x3

    modules = list(model.named_modules())
    for i in range(len(modules) - 1):
        name1, conv = modules[i]
        name2, bn = modules[i + 1]

        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            with torch.no_grad():
                W = conv.weight       # [out_ch, in_ch, kH, kW]
                out_ch, in_ch, kH, kW = W.shape

                # --- svd sum ------
                W_flat = W.view(out_ch, -1)
                # W_flat = W.view(W.size()[0], W.size()[1], -1).sum(dim=2)

                U, S, Vh = torch.svd(W_flat)
                # print(U.size(), S.size(), Vh.size(), W.size())


                # ------ adaptive kepp_ratio ----------
                # n = torch.sum(S)
                # cs = torch.cumsum(S, 0)
                # n = torch.norm(S)
                # cs = torch.sqrt(torch.cumsum(S ** 2, 0))

                # remain = n * keep_ratio
                # m = cs <= remain

                # rank = len(m[m == True])
                rank = max(1, int(len(S) * keep_ratio))

                # ---- without low-rank ------------
                U_r = U[:, :rank]
                S_r = S[:rank]
                Vh_r = Vh[:, :rank]
                # U_r = U
                # S_r = S
                # Vh_r = Vh

                center = bn.running_mean

                W_eff = W.view(W.size()[0], W.size()[1], -1).sum(dim=2)  # shape: [out_ch, in_ch]
                W_pinv = torch.pinverse(W_eff)        # [rc, out_ch]
                center_before = W_pinv @ center

                dotp = torch.abs(U_r.T @ center)
                # print(center.size(), center_before.size(), U.size(), Vh.size(), U_r.size(), Vh_r.size())
                # dotp = torch.abs(Vh_r.T @ center_before)
                correlated   = torch.nonzero(dotp >= S_r[-1], as_tuple=True)[0]
                uncorrelated = torch.nonzero(dotp <  S_r[-1], as_tuple=True)[0]
                # correlated   = torch.nonzero(dotp >= S_r[rank - 1], as_tuple=True)[0]
                # uncorrelated = torch.nonzero(dotp <  S_r[rank - 1], as_tuple=True)[0]

                print(len(correlated), len(uncorrelated), rank, len(S) - rank)

                # -------- helper для сборки пути --------
                def make_path(idx):
                    if idx.numel() == 0:
                        return None
                    U_sel = U_r[:, idx]                    # [out_ch, rc]
                    S_sel = S_r[idx]                       # [rc]
                    V_sel = Vh_r[:, idx]                    # [in_dim, rc]

                    # первый conv
                    W1 = V_sel.T.contiguous().view(idx.numel(),
                                                   in_ch // conv.groups,
                                                   kH, kW)  # [rc, in_ch/groups, kH, kW]

                    # второй conv (1x1)
                    W2 = (U_sel * S_sel).contiguous().view(out_ch, idx.numel(), 1, 1)

                    # центр до conv2
                    W_eff = W2.view(out_ch, idx.numel())  # [out_ch, rc]
                    W_pinv = torch.pinverse(W_eff)        # [rc, out_ch]
                    center_before = W_pinv @ center       # [rc]
                    return W1, W2, center_before

                corr_pack   = make_path(correlated)
                uncorr_pack = make_path(uncorrelated)

                if corr_pack is None:
                    conv._has_corr = False
                else:
                    conv._has_corr = True
                    W1_corr, W2_corr, c_corr = corr_pack
                    conv.register_buffer("_W1_corr", W1_corr)
                    conv.register_buffer("_W2_corr", W2_corr)
                    conv.register_buffer("_center_corr", c_corr)

                if uncorr_pack is None:
                    conv._has_uncorr = False
                else:
                    conv._has_uncorr = True
                    W1_uncorr, W2_uncorr, c_uncorr = uncorr_pack
                    conv.register_buffer("_W1_uncorr", W1_uncorr)
                    conv.register_buffer("_W2_uncorr", W2_uncorr)
                    conv.register_buffer("_center_uncorr", c_uncorr)


def svd_forward_hook_conv_pre(name, bn_layer):
    """
    Хук для Conv2d: аппроксимация весов через усечённое SVD.
    keep_ratio — доля сингулярных компонент, которые нужно оставить (0..1).
    """
    def hook_fn(module, input, output):
        with torch.no_grad():
            center = bn_layer.running_mean
            x = input[0]
            y = None

            if getattr(module, "_has_corr", False):
                y_corr_hidden = F.conv2d(
                    x, module._W1_corr, None,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups
                )
                if module._center_corr is not None:
                    y_corr_hidden = y_corr_hidden - module._center_corr.view(1, -1, 1, 1)

                y_corr = F.conv2d(y_corr_hidden, module._W2_corr, None,
                                  stride=1, padding=0, dilation=1, groups=1)
                y = y_corr if y is None else (y + y_corr)

            if getattr(module, "_has_uncorr", False):
                y_uncorr_hidden = F.conv2d(
                    x, module._W1_uncorr, None,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups
                )
                # центр можно тоже вычесть, если нужно
                # if module._center_uncorr is not None:
                #     y_uncorr_hidden = y_uncorr_hidden - module._center_uncorr.view(1, -1, 1, 1)

                y_uncorr = F.conv2d(y_uncorr_hidden, module._W2_uncorr, None,
                                    stride=1, padding=0, dilation=1, groups=1)
                y = y_uncorr if y is None else (y + y_uncorr)

            if y is None:
                # fallback: слой занулен
                y = F.conv2d(x, module.weight*0, None,
                             stride=module.stride, padding=module.padding,
                             dilation=module.dilation, groups=module.groups)

            if module.bias is not None:
                y = y + module.bias.view(1, -1, 1, 1)

            # BN scale
            # y = y - bn_layer.running_mean.view(1, -1, 1, 1)
            y = y / torch.sqrt(bn_layer.running_var.view(1, -1, 1, 1) + bn_layer.eps)
            return y

    return hook_fn


def register_svd_hooks(model, layers, keep_ratio):
    remove_all_hooks(model)
    modules = list(model.named_modules())

    precompute_svd_weights(model, keep_ratio)

    for i in range(len(modules) - 1):
        name1, layer1 = modules[i]
        name2, layer2 = modules[i + 1]

        # if name1 in layers:
        if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.BatchNorm2d):
            layer1.register_forward_hook(svd_forward_hook_conv_pre(name1, layer2))
            layer2.register_forward_hook(hook_bn_modified(name2))
            # layer1.register_forward_hook(lambda module, input, output, name=name1: hook_conv(name, module, input, output))
            # layer2.register_forward_hook(lambda module, input, output, name=name2: hook_bn(name, module, input, output))
    # return bns, convs



@logger.catch
def main():
    args = make_parser().parse_args()

    torch.cuda.set_device(args.local_rank)

    device = 'cuda'
    config = parse_config(args.config_file)
    if args.opts:
        config.merge(args.opts)

    save_dir = os.path.join(config.miscs.output_dir, config.miscs.exp_name)
    if args.local_rank == 0:
        mkdir(save_dir)
    setup_logger(save_dir, distributed_rank=args.local_rank, mode='w')

    # --- Model ---
    model = build_local_model(config, device)
    model.head.nms = True

    model.cuda(args.local_rank)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    logger.info(f'Loaded ckpt from: {args.ckpt}')

    if not config.test.bn_track_running_stats:
        to_del = []
        for k, v in ckpt['model'].items():
            if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                # del ckpt['model'][k]
                to_del.append(k)
        for k in to_del:
            del ckpt['model'][k]

    new_state_dict = {}
    for k, v in ckpt['model'].items():
        k = k.replace('module', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    logger.info('loaded checkpoint done.')

    # for layer in model.modules():
    #     if isinstance(layer, RepConv):
    #         layer.switch_to_deploy()

    # ==== hooks and change order of operations ====
    register_conv_bn_hooks_modified(model)
    # ==============================================

    # ==== hooks for svd ============
    # layers_to_add = ['backbone.block_list.3.block_list.0.conv1.conv1']
    # layers_to_add = ['backbone.block_list.3.block_list.0.conv2.rbr_dense.conv']
    # layers_to_add += ['neck.merge_7.conv1.conv']

    # register_svd_hooks(model, layers_to_add, 0.95)
    # ===============================

    if args.fuse:
        logger.info('\tFusing model...')
        model = fuse_model(model)

    # --- Dataset(s) → alphabetical subset → DataLoader(s) ---
    raw_val_datasets = build_dataset(config, config.dataset.val_ann, is_train=False)
    val_datasets = []
    for ds, ann_name in zip(raw_val_datasets, config.dataset.val_ann):
        logger.info(f'Preparing subset for dataset: {ann_name} (len={len(ds)})')
        val_datasets.append(build_alphabetical_subset(ds, args.subset_size))

    val_loaders = build_dataloader(
        val_datasets,
        config.test.augment,
        batch_size=args.batch_size,
        num_workers=config.miscs.num_workers,
        is_train=False,
        size_div=32,
    )

    # --- Evaluate ---
    output_folders = [None] * len(config.dataset.val_ann)
    if args.local_rank == 0 and config.miscs.output_dir:
        for idx, dataset_name in enumerate(config.dataset.val_ann):
            output_folder = os.path.join(config.miscs.output_dir, 'inference', dataset_name + f'_subset{args.subset_size}')
            mkdir(output_folder)
            output_folders[idx] = output_folder

    for output_folder, dataset_name, data_loader_val in zip(output_folders, config.dataset.val_ann, val_loaders):
        logger.info(f'\n>>> Start evaluation on subset for `{dataset_name}`')
        inference(
            model,
            data_loader_val,
            dataset_name + f'_subset{args.subset_size}',
            iou_types=('bbox',),
            box_only=False,
            device=device,
            output_folder=output_folder,
        )


if __name__ == '__main__':
    main()
