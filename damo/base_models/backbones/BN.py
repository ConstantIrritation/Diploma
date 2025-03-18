# from ChatGPT

import torch
import torch.nn as nn


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if not self.track_running_stats:
            mean = torch.zeros(num_features).to(device)
            var =  torch.ones(num_features).to(device)

        # Learnable affine parameters
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))  # Scale factor
            self.bias = nn.Parameter(torch.zeros(num_features))  # Shift factor
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Running statistics for inference
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Expected 4D input (N, C, H, W), got {}".format(x.shape))

        N, C, H, W = x.shape
        assert C == self.num_features, "Expected {} channels, but got {}".format(self.num_features, C)

        if self.training or not self.track_running_stats:
            # Compute batch statistics
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # Mean over N, H, W
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)  # Variance over N, H, W

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
                    self.num_batches_tracked += 1
        else:
            # Use running statistics for inference
            mean = self.running_mean.view(1, C, 1, 1)
            var = self.running_var.view(1, C, 1, 1)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable parameters
        if self.affine:
            x_norm = self.weight.view(1, C, 1, 1) * x_norm + self.bias.view(1, C, 1, 1)

        return x_norm

'''
#from https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py

def compare_bn(bn1, bn2):
    err = False
    if not torch.allclose(bn1.running_mean, bn2.running_mean):
        print('Diff in running_mean: {} vs {}'.format(
            bn1.running_mean, bn2.running_mean))
        err = True

    if not torch.allclose(bn1.running_var, bn2.running_var):
        print('Diff in running_var: {} vs {}'.format(
            bn1.running_var, bn2.running_var))
        err = True

    if bn1.affine and bn2.affine:
        if not torch.allclose(bn1.weight, bn2.weight):
            print('Diff in weight: {} vs {}'.format(
                bn1.weight, bn2.weight))
            err = True

        if not torch.allclose(bn1.bias, bn2.bias):
            print('Diff in bias: {} vs {}'.format(
                bn1.bias, bn2.bias))
            err = True

    if not err:
        print('All parameters are equal!')

torch.manual_seed(42)
m = nn.BatchNorm2d(100)
inp = torch.randn(20, 100, 35, 45)
out1 = m(inp)
n = MyBatchNorm2d(100)
out2 = n(inp)
# out1 == out2
compare_bn(m, n)

torch.allclose(out1, out2)
print('Max diff: ', (out1 - out2).abs().max())

All parameters are equal!
Max diff:  tensor(4.7684e-07, grad_fn=<MaxBackward1>)
'''