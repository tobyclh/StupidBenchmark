"""this script tests how much performance gain does nn.Sequential grant us.
"""
import torch
from torch import nn


seq = nn.Sequential(
            nn.Conv1d(in_channels=1,    out_channels=16,    kernel_size=250, stride=50, padding=110),
            nn.ReLU(),
            nn.Conv1d(in_channels=16,   out_channels=32,    kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32,   out_channels=64,    kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64,   out_channels=128,   kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128,  out_channels=256,   kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256,  out_channels=512,   kernel_size=4, stride=2, padding=1),
            nn.ReLU())

def discrete():

    x1 = nn.Conv1d(in_channels=1,    out_channels=16,    kernel_size=250, stride=50, padding=110)
    x2 = nn.ReLU()
    x3 = nn.Conv1d(in_channels=16,   out_channels=32,    kernel_size=4, stride=2, padding=2)
    x4 = nn.ReLU()
    x5 = nn.Conv1d(in_channels=32,   out_channels=64,    kernel_size=4, stride=2, padding=1)
    x6 = nn.ReLU()
    x7 = nn.Conv1d(in_channels=64,   out_channels=128,   kernel_size=4, stride=2, padding=2)
    x8 = nn.ReLU()
    x9 = nn.Conv1d(in_channels=128,  out_channels=256,   kernel_size=4, stride=2, padding=1)
    x10 = nn.ReLU()
    x11 = nn.Conv1d(in_channels=256,  out_channels=512,   kernel_size=4, stride=2, padding=1)
    x12 = nn.ReLU()
    def forward(input):
        return x12(x11(x10(x9(x8(x7(x6(x5(x4(x3(x2(x1(input))))))))))))
    return forward

import torch
from skimage import io, transform
from time import time
from torch import nn
from time import time
from torchvision.transforms.functional import resize
from PIL import Image
from torch.nn.functional import upsample

def timereps(reps, func):
    start = time()
    [func() for _ in range(0, reps)]
    end = time()
    return (end - start) / reps

forward_discrete = discrete()

x = torch.randn(10, 1, 8000)
average_duration = timereps(1000, lambda : seq(x))
print(f'Sequential (CPU): {average_duration}')

average_duration = timereps(1000, lambda : forward_discrete(x))
print(f'Discrete (CPU): {average_duration}')

"""
Conclusion : barely any difference
Edit : 
this is justified by the fact that nn.Sequential is barely a syntactic sugar, unless they actually start to compile it it wouldn't make any different
https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
"""



