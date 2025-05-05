# remove all
# https://discuss.pytorch.org/t/how-do-i-remove-forward-hooks-on-a-module-without-the-hook-handles/140393


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
            elif hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict() # type: ignore
            elif hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict() # type: ignore
            remove_all_hooks(child)


def hook_fn_fwd(module, input, output):
    print('forward hook used')
    global layer_outputs_fwd
    layer_outputs_fwd[module] = [input[0].squeeze(0), output.squeeze(0)]

def hook_fn_bck(module, input, output):
    print('backward hook used')
    global layer_outputs_bck
    layer_outputs_bck[module] = [input, output]


def register_hooks(model, layers, backward=False):
    remove_all_hooks(model)
    for layer_to_add in layers:
        for name, layer in model.named_modules():
            if layer_to_add in name:
                print(f'add {layer_to_add}')
                layer.register_forward_hook(hook_fn_fwd)
                if backward:
                    layer.register_backward_hook(hook_fn_bck)
    if backward:
        return layer_outputs_fwd, layer_outputs_bck
    else:
        return layer_outputs_fwd


def hook_bn_noname(module, input, output):
    global bns
    bns.append([module, input[0].squeeze(0), output.squeeze(0)])


def register_bn_hooks(model):
    remove_all_hooks(model)
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.register_forward_hook(hook_bn_noname)
    return bns


def hook_fn_fwd_batch(module, input, output):
    print('forward hook used')
    global layer_outputs_fwd
    layer_outputs_fwd[module] = [input[0], output]


def register_hooks_batch_forward(model, layers):
    remove_all_hooks(model)
    for layer_to_add in layers:
        for name, layer in model.named_modules():
            if layer_to_add in name:
                layer.register_forward_hook(hook_fn_fwd_batch)
    return layer_outputs_fwd

def hook_conv(name, module, input, output):
    global convs
    convs.append([name, module, input[0].squeeze(0), output.squeeze(0)])

def hook_bn(name, module, input, output):
    global bns
    bns.append([name, module, input[0].squeeze(0), output.squeeze(0)])

def register_conv_bn_hooks(model):
    remove_all_hooks(model)
    modules = list(model.named_modules())
    for i in range(len(modules) - 1):
        name1, layer1 = modules[i]
        name2, layer2 = modules[i + 1]

        if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.BatchNorm2d):
            layer1.register_forward_hook(lambda module, input, output, name=name1: hook_conv(name, module, input, output))
            layer2.register_forward_hook(lambda module, input, output, name=name2: hook_bn(name, module, input, output))
    return bns, convs