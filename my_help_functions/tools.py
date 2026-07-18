from functools import reduce

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from tqdm import tqdm

from damo.config.base import parse_config
from my_help_functions.cosine_matrix import (
    get_positions_of_classes_on_flattened_image_for_collage,
)
from my_help_functions.hooks import register_conv_bn_hooks, register_hooks
from tools.demo import Infer


def cos_sim(x):
    return cosine_similarity(x, x)


def pairwise_cosine_similarity(x1):
    x1 = F.normalize(x1, dim=0)
    return x1.T @ x1


def load_model():
    config = parse_config('./configs/damoyolo_tinynasL20_T.py')
    infer_engine = Infer(config, device='cuda',
        ckpt='./weights/damoyolo_tiny.pth')
    model = infer_engine.model.eval()
    return infer_engine, model


def load_collage(idx, gray):
    if gray:
        if idx not in (5, 7, 9):
            raise ValueError("currently available only 5, 7, 9 idxs for gray bakground collage")
    else:
        if idx not in (1, 2, 5, 19, 25, 36):
            raise ValueError("currently available only 1, 2, 5, 19, 25, 36 idxs for image bakground collage")

    path = f"./collage/{'gray_' if gray else ''}collage_{idx}.jpg"

    origin_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return origin_image


def get_feature_maps(infer_engine, model, conv_path, bn_path, origin_image):

    layers_to_add = [conv_path, bn_path]
    bn = reduce(getattr, bn_path.split("."), model)

    layer_outputs_fwd = register_hooks(model, layers_to_add)

    image, origin_shape = infer_engine.preprocess(origin_image)
    output = model(image)

    data = [(layers_to_add[i], layer, str(embeddings[0].shape), str(embeddings[1].shape)) for i, (layer, embeddings) in enumerate(layer_outputs_fwd.items())]
    print(tabulate(data, headers=["Название", "Слой", "Размер входа", "Размер выхода"], tablefmt="grid"))
    
    conv = {}
    conv['before conv'] = list(layer_outputs_fwd.values())[0][0]
    conv['after conv']  = list(layer_outputs_fwd.values())[0][1]
    conv['after center'] = list(layer_outputs_fwd.values())[1][0] - bn.running_mean.view(-1, 1, 1)
    # center = bn.running_mean.view(-1, 1, 1)

    to_create = ['before conv', 'after conv', 'after center']
    matrices = {}
    for name in to_create:
        vec = conv[name]
        reshaped = vec.flatten(1, 2).permute(1, 0).detach().cpu().numpy()
    #     print(reshaped.shape)
        matrices[name] = reshaped
    
    return conv, matrices


def get_feature_maps_for_all_layers(infer_engine, model, origin_image):
    bns, convs = register_conv_bn_hooks(model)

    image, origin_shape = infer_engine.preprocess(origin_image)
    output = model(image)

    before_conv = np.array([convs[i][2].detach().cpu() for i in range(len(bns))], dtype=object)
    after_conv = np.array([convs[i][3].detach().cpu() for i in range(len(bns))], dtype=object)

    after_center = np.array([bns[i][2].detach().cpu()
                        - bns[i][1].running_mean.view(-1, 1, 1).detach().cpu() for i in range(len(bns))], dtype=object)
    names = np.array([convs[i][0] for i in range(len(bns))], dtype=object)

    data = [(i + 1, before_conv[i].shape, after_conv[i].shape, after_center[i].shape, names[i]) for i in range(len(bns))]
    print(tabulate(data, headers=["Layer", "before_conv size", "after_conv size", "after_center size", 'name'], tablefmt="grid"))

    bns_fwd_names = [name for name, _, _, _ in bns]

    return before_conv, after_conv, after_center, bns_fwd_names


def get_angles_all_model(
    before_conv,
    after_conv,
    after_center,
    idx,
    gray,
    type,          # while | after
    operation,     # conv | center
):
    inside = []
    outside = []
    back = []
    inside_all = []
    outside_all = []
    back_all = []


    for k in tqdm(range(2, len(before_conv))):
        if type == "while":
            position_source = before_conv if operation == "conv" else after_center
        else:
            position_source = after_conv if operation == "conv" else after_center

        positions, class_names = get_positions_of_classes_on_flattened_image_for_collage(
            idx,
            position_source[k].size()[1],
            f"{'gray_' if gray else ''}"
        )
        if type == "while":

            if operation == "conv":
                if before_conv[k].size()[1] != after_conv[k].size()[1]:
                    u = torch.nn.Upsample(scale_factor=2, mode="bilinear")
                    a = u(after_conv[k].unsqueeze(0)).squeeze()
                    matr1 = a.flatten(1, 2).detach()
                else:
                    matr1 = after_conv[k].flatten(1, 2).detach()

                matr2 = before_conv[k].flatten(1, 2).detach()

            else:  # center
                matr1 = after_center[k].flatten(1, 2).detach()
                matr2 = after_conv[k].flatten(1, 2).detach()

            csm = pairwise_cosine_similarity(matr1) - pairwise_cosine_similarity(matr2)

        else:  # after

            matr = (
                after_conv[k].flatten(1, 2).detach()
                if operation == "conv"
                else after_center[k].flatten(1, 2).detach()
            )

            csm = pairwise_cosine_similarity(matr)

        n = len(positions)

        mean_angle_change_w_others_list = []
        angle_change_w_self_list = []
        angle_change_w_back_list = []

        for i in range(n - 1):

            mean_angle_change_w_others = 0
            
            for j in range(n):
                pos_i = torch.tensor(positions[i + 1], device=csm.device)
                pos_j = torch.tensor(positions[j + 1], device=csm.device)
                pair_matrix = torch.meshgrid(pos_i, pos_j)
                submatrix = csm[pair_matrix[0], pair_matrix[1]]

                mean = submatrix.mean()
                
                if i == j:
                    angle_change_w_self_list.append(mean)
                elif j == (n - 1):
                    angle_change_w_back_list.append(mean)
                else:
                    mean_angle_change_w_others += mean
            
            mean_angle_change_w_others /= (n - 2)
            mean_angle_change_w_others_list.append(mean_angle_change_w_others)

        inside.append(torch.tensor(angle_change_w_self_list).mean())
        outside.append(torch.tensor(mean_angle_change_w_others_list).mean())
        back.append(torch.tensor(angle_change_w_back_list).mean())
        inside_all.append(torch.tensor(angle_change_w_self_list))
        outside_all.append(torch.tensor(mean_angle_change_w_others_list))
        back_all.append(torch.tensor(angle_change_w_back_list))

    inside_np = torch.stack(inside_all).cpu().numpy()
    outside_np = torch.stack(outside_all).cpu().numpy()
    back_np = torch.stack(back_all).cpu().numpy()

    return inside, outside, back, inside_np, outside_np, back_np, class_names