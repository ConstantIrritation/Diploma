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

# from my_help_functions.hooks import remove_all_hooks
import torch.nn.functional as F

from collections import OrderedDict
from typing import Dict, Callable
import torch
import torch.nn as nn

layer_outputs_fwd = {}
layer_outputs_bck = {}
bns = []
convs = []

def remove_all_hooks(model: torch.nn.Module) -> None:
    global layer_outputs_fwd
    layer_outputs_fwd = {}
    global layer_outputs_bck
    layer_outputs_bck = {}
    global bns
    bns = []
    global convs
    convs = []

    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict() # type: ignore
            if hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict() # type: ignore
            if hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict() # type: ignore
            remove_all_hooks(child)

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


def hook_bn_affine(name):
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

                W = conv.weight       # [out_ch, in_ch, kH, kW]
                out_ch, in_ch, kH, kW = W.shape
                center = bn.running_mean

                W_eff = W.view(W.size()[0], W.size()[1], -1).sum(dim=2)  # shape: [out_ch, in_ch]
                W_pinv = torch.pinverse(W_eff)        # [in_ch, out_ch]
                center_before = W_pinv @ center

                # W_fl = W.flatten(1)
                # W_pinv = torch.pinverse(W_fl)        # [k * k * in_ch, out_ch]
                # center_before = W_pinv @ center
                # center_before = center_before.view(in_ch, kH * kW).mean(1)

                # W_fl = W.flatten(1)
                # U, S, Vh = torch.svd(W_fl)
                # U_inv = torch.pinverse(U)
                # Vh_inv = torch.pinverse(Vh)
                # Vh_inv = Vh_inv.view(Vh_inv.size()[0], in_ch, kH, kW).mean(dim=(2, 3)).permute(1, 0)
                # center_before_U = U_inv @ center
                # center_before_S = center_before_U / S
                # center_before = Vh_inv @ center_before_S

                conv.register_buffer("_pre_center", center_before)


def hook_conv_change_conv_center(name, bn_layer):
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

            # BN center norm
            # y = y - bn_layer.running_mean.view(1, -1, 1, 1)
            y = y / torch.sqrt(bn_layer.running_var.view(1, -1, 1, 1) + bn_layer.eps)
            return y

    return hook_fn

def register_hooks_change_conv_center(model):
    remove_all_hooks(model)
    modules = list(model.named_modules())

    precompute_pre_center(model)

    for i in range(len(modules) - 1):
        name1, layer1 = modules[i]
        name2, layer2 = modules[i + 1]
        
        if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.BatchNorm2d):
            layer1.register_forward_hook(hook_conv_change_conv_center(name1, layer2))
            layer2.register_forward_hook(hook_bn_affine(name2))


def precompute_svd_weights(model, keep_ratio):
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

                # print(len(correlated), len(uncorrelated), rank, len(S) - rank)

                # -------- helper для сборки пути --------
                def make_path(idx):
                    if idx.numel() == 0:
                        return None
                    U_sel = U_r[:, idx]  # [out_ch, rc]
                    S_sel = S_r[idx]     # [rc]
                    V_sel = Vh_r[:, idx] # [in_dim, rc]

                    W1 = V_sel.T.contiguous().view(idx.numel(),
                                                   in_ch // conv.groups,
                                                   kH, kW)  # [rc, in_ch/groups, kH, kW]

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

                conv.register_buffer("_correlated_idx", correlated)
                conv.register_buffer("_uncorrelated_idx", uncorrelated)


def hook_conv_corr_uncorr(name, bn_layer):
    def hook_fn(module, input, output):
        with torch.no_grad():
            global layer_outputs_fwd

            center = bn_layer.running_mean
            x = input[0]
            y = None

            conv_orig = output.detach().cpu() if isinstance(output, torch.Tensor) else None
            rec = {
                "conv_out": conv_orig,
                "bn_center": center.detach().cpu(),
                }


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

                rec.update({
                    "center_corr": module._center_corr.detach().cpu() if isinstance(module._center_corr, torch.Tensor) else None,
                    "y_corr_hidden": y_corr_hidden.detach().cpu(),
                    "y_corr": y_corr.detach().cpu(),
                })

            if getattr(module, "_has_uncorr", False):
                y_uncorr_hidden = F.conv2d(
                    x, module._W1_uncorr, None,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups
                )
                if module._center_uncorr is not None:
                    y_uncorr_hidden = y_uncorr_hidden - module._center_uncorr.view(1, -1, 1, 1)

                y_uncorr = F.conv2d(y_uncorr_hidden, module._W2_uncorr, None,
                                    stride=1, padding=0, dilation=1, groups=1)
                y = y_uncorr if y is None else (y + y_uncorr)

                rec.update({
                    "center_uncorr": module._center_uncorr.detach().cpu() if isinstance(module._center_uncorr, torch.Tensor) else None,
                    "y_uncorr_hidden": y_uncorr_hidden.detach().cpu(),
                    "y_uncorr": y_uncorr.detach().cpu(),
                })

            if module.bias is not None:
                y = y + module.bias.view(1, -1, 1, 1)

            # BN center norm
            # y = y - bn_layer.running_mean.view(1, -1, 1, 1)
            y = y / torch.sqrt(bn_layer.running_var.view(1, -1, 1, 1) + bn_layer.eps)

            if hasattr(module, "_correlated_idx"):
                try:
                    rec["correlated_idx"] = module._correlated_idx.detach().cpu()
                except Exception:
                    rec["correlated_idx"] = module._correlated_idx

            if hasattr(module, "_uncorrelated_idx"):
                try:
                    rec["uncorrelated_idx"] = module._uncorrelated_idx.detach().cpu()
                except Exception:
                    rec["uncorrelated_idx"] = module._uncorrelated_idx

            layer_outputs_fwd[name] = rec

            return y

    return hook_fn


# def hook_conv_corr_uncorr(name, bn_layer, keep_ratio):
    # def hook_fn(module, input, output):
    #     with torch.no_grad():
    #         x = input[0]                              # [N, C_in, H, W]
    #         device = x.device

    #         # --- get precomputed SVD buffers or fallback ---
    #         if hasattr(module, "_W_flat") and hasattr(module, "_Vh_r"):
    #             W_flat = module._W_flat              # [out_ch, in_dim]
    #             Vh_r = module._Vh_r                 # [in_dim, rank]
    #             rank = module._svd_rank.item()
    #             U_r = module._U_r if hasattr(module, "_U_r") else None
    #             S_r = module._S_r if hasattr(module, "_S_r") else None
    #         else:
    #             W = module.weight
    #             out_ch, in_ch, kH, kW = W.shape
    #             W_flat = W.view(out_ch, -1)
    #             U, S, Vh = torch.svd(W_flat)
    #             rank = max(1, int(len(S) * keep_ratio))
    #             Vh_r = Vh[:, :rank].to(device)
    #             U_r = U[:, :rank].to(device)
    #             S_r = S[:rank].to(device)

    #         out_ch, in_dim = W_flat.shape
    #         N, C_in, H_in, W_in = x.shape
    #         kH, kW = module.kernel_size

    #         # --- unfold input to patches ---
    #         unfold = torch.nn.Unfold(kernel_size=(kH, kW),
    #                                  dilation=module.dilation,
    #                                  padding=module.padding,
    #                                  stride=module.stride)
    #         X = unfold(x)                     # [N, in_dim, L]
    #         N, in_dim2, L = X.shape
    #         assert in_dim2 == in_dim

    #         # --- project patches onto span(Vh_r) and compute orthogonal part ---
    #         X_mat = X.permute(1, 0, 2).reshape(in_dim, -1)   # [in_dim, N*L]
            
    #         # Коэффициенты проекции на span(Vh_r)
    #         c = Vh_r.t() @ X_mat                             # [rank, N*L]
            
    #         # ========== ПРАВИЛЬНОЕ ВЫЧИСЛЕНИЕ СВЕРТКИ ==========
    #         # Используем факторизованную форму: W_r = U_r @ diag(S_r) @ Vh_r.T
    #         # Тогда: W_r @ X = U_r @ diag(S_r) @ (Vh_r.T @ X) = U_r @ diag(S_r) @ c
            
    #         Y_proj_mat = U_r @ (S_r.unsqueeze(1) * c)       # [out_ch, N*L]
            
    #         # Для полной свертки тоже используем факторизацию (при keep_ratio=1.0 должно совпадать)
    #         # Но для диагностики посчитаем через W_flat
    #         Y_full_mat = W_flat @ X_mat                      # [out_ch, N*L]
            
    #         # Ортогональная компонента: применяем W_r к ортогональной части входа
    #         # W_r @ X_orth = U_r @ diag(S_r) @ (Vh_r.T @ X_orth)
    #         # Но Vh_r.T @ X_orth ≈ 0 по построению!
    #         c_orth = Vh_r.t() @ (X_mat - Vh_r @ c)           # должно быть ≈ 0
    #         Y_orth_mat = U_r @ (S_r.unsqueeze(1) * c_orth)   # должно быть ≈ 0
            
    #         # ДИАГНОСТИКА 1: проверяем ортогональность
    #         X_proj_mat = Vh_r @ c
    #         X_orth_mat = X_mat - X_proj_mat
    #         inner_product = (X_proj_mat * X_orth_mat).sum(dim=0).mean().cpu().item()
            
    #         # ДИАГНОСТИКА 2: проверяем, что Vh_r^T @ X_orth ≈ 0
    #         c_orth_norm = c_orth.abs().mean().cpu().item()
            
    #         # Проверка реконструкции: при keep_ratio=1.0 должно быть Y_proj ≈ Y_full
    #         Y_sum_mat = Y_proj_mat + Y_orth_mat
    #         reconstruction_error = (Y_full_mat - Y_proj_mat).abs().mean().cpu().item()
    #         sum_reconstruction_error = (Y_full_mat - Y_sum_mat).abs().mean().cpu().item()
            
    #         # Reshape обратно
    #         Y_full = Y_full_mat.reshape(out_ch, N, L).permute(1, 0, 2).contiguous()
    #         Y_proj = Y_proj_mat.reshape(out_ch, N, L).permute(1, 0, 2).contiguous()
    #         Y_orth = Y_orth_mat.reshape(out_ch, N, L).permute(1, 0, 2).contiguous()

    #         # --- fold patch-outputs into spatial maps ---
    #         H_out = (H_in + 2*module.padding[0] - module.dilation[0]*(kH-1) - 1)//module.stride[0] + 1
    #         W_out = (W_in + 2*module.padding[1] - module.dilation[1]*(kW-1) - 1)//module.stride[1] + 1
    #         fold_out = torch.nn.Fold(output_size=(H_out, W_out), kernel_size=(1,1), stride=1)

    #         y_full = fold_out(Y_full.permute(0,2,1).reshape(N, out_ch, L))
    #         y_proj = fold_out(Y_proj.permute(0,2,1).reshape(N, out_ch, L))
    #         y_orth = fold_out(Y_orth.permute(0,2,1).reshape(N, out_ch, L))

    #         # --- add bias ---
    #         if module.bias is not None:
    #             b = module.bias.view(1, -1, 1, 1).to(device)
    #             y_full = y_full + b
    #             y_proj = y_proj + b
    #             y_orth = y_orth + b

    #         # --- МЕТРИКИ БЕЗ ЦЕНТРИРОВАНИЯ (для чистой диагностики) ---
    #         norm_orth_raw = y_orth.abs().mean().cpu().item()
    #         norm_full_raw = y_full.abs().mean().cpu().item()
    #         norm_proj_raw = y_proj.abs().mean().cpu().item()
    #         diff_raw = (y_full - y_proj - y_orth).abs().mean().cpu().item()
            
    #         # --- CENTERING: только для y_proj! ---
    #         # y_full = y_proj + y_orth
    #         # y_full - center = (y_proj - center) + y_orth
    #         # Поэтому центрируем только y_proj, y_orth остается как есть!
    #         center = bn_layer.running_mean.to(device).view(1, -1, 1, 1)
    #         y_full_c  = y_full  - center
    #         y_proj_c  = y_proj  - center
    #         y_orth_c  = y_orth  # НЕ вычитаем center!

    #         # --- МЕТРИКИ ПОСЛЕ ЦЕНТРИРОВАНИЯ ---
    #         norm_orth_centered = y_orth_c.abs().mean().cpu().item()
    #         diff_after_centering = (y_full_c - y_proj_c - y_orth_c).abs().mean().cpu().item()
            
    #         print(f"[{name}] rank={rank}, in_dim={in_dim}, out_ch={out_ch}")
    #         print(f"  Проекция: <X_proj,X_orth>={inner_product:.3e}, |Vh_r^T*X_orth|={c_orth_norm:.3e}")
    #         print(f"  Свертка (raw): |Y_orth|={norm_orth_raw:.3e}, |Y_proj|={norm_proj_raw:.3e}, |Y_full|={norm_full_raw:.3e}")
    #         print(f"  Разность: |Y_full - Y_proj|={reconstruction_error:.3e}, |Y_full - Y_proj - Y_orth|={sum_reconstruction_error:.3e}")
    #         print(f"  Разность: |Y_full - Y_proj|={(y_full - y_proj).abs().mean().cpu().item():.3e}")
    #         print(f"  После центрир: |Y_orth|={norm_orth_centered:.3e}, |Y_proj_c|={y_proj_c.abs().mean().cpu().item():.3e}")
    #         print(f"  Проверка: |Y_full_c - Y_proj_c - Y_orth_c|={diff_after_centering:.3e}")
            
    #         # --- final output: возвращаем нормализованный y_proj ---
    #         y = y_proj_c
    #         y = y / torch.sqrt(bn_layer.running_var.view(1, -1, 1, 1).to(device) + bn_layer.eps)
            
    #         # ДИАГНОСТИКА: сравним с оригинальным output
    #         original_output = output
    #         print(f"  Output сравнение: |original|={original_output.abs().mean().cpu().item():.3e}, |modified|={y.abs().mean().cpu().item():.3e}")
    #         print(f"  Difference: |original - modified|={(original_output - y).abs().mean().cpu().item():.3e}")

    #         # cleanup
    #         del X, X_mat, X_proj_mat, X_orth_mat, c, c_orth
    #         del Y_full_mat, Y_proj_mat, Y_orth_mat, Y_sum_mat
    #         del Y_full, Y_proj, Y_orth
    #         del y_full, y_proj, y_orth, y_full_c, y_proj_c, y_orth_c
    #         torch.cuda.empty_cache()

    #         return y

    # return hook_fn


def register_hooks_corr_uncorr(model, layers, keep_ratio):
    remove_all_hooks(model)
    modules = list(model.named_modules())

    precompute_svd_weights(model, keep_ratio)

    for i in range(len(modules) - 1):
        name1, layer1 = modules[i]
        name2, layer2 = modules[i + 1]

        if name1 in layers:
        # if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.BatchNorm2d):
            layer1.register_forward_hook(hook_conv_corr_uncorr(name1, layer2))
            layer2.register_forward_hook(hook_bn_affine(name2))

    return layer_outputs_fwd


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

    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()

    # ==== hooks and change order of operations ====
    # register_hooks_change_conv_center(model)
    # ==============================================

    # ==== hooks for svd ============
    layers_to_add = ['backbone.block_list.3.block_list.0.conv1.conv1']
    # layers_to_add = ['backbone.block_list.3.block_list.0.conv2.rbr_dense.conv']
    # layers_to_add += ['neck.merge_7.conv1.conv']

    register_hooks_corr_uncorr(model, layers_to_add, 1.)
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
