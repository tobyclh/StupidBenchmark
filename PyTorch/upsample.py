import torch
from skimage import io, transform
from time import time
from torch import nn
from time import time
from torchvision.transforms.functional import resize
from PIL import Image
from torch.nn.functional import upsample
img = io.imread('data/britney.png')
def timereps(reps, func):
    start = time()
    for i in range(0, reps):
        func()
    end = time()
    return (end - start) / reps

average_duration = timereps(10, lambda : transform.rescale(img, [2,2], mode='constant'))
print(f'transform.rescale : {average_duration}')


tensor_img = torch.Tensor(img)
tensor_img = tensor_img.cuda().unsqueeze(0)
tensor_img = tensor_img.expand(1, -1, -1, -1).permute(0, 3, 1, 2)
up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

average_duration = timereps(1000, lambda : up(tensor_img))
print(f'nn.Upsample (GPU): {average_duration}')

average_duration = timereps(1000, lambda : upsample(tensor_img, scale_factor=2, mode='bilinear', align_corners=False))
print(f'nn.functional.upsample (GPU) : {average_duration}')

tensor_img = tensor_img.cpu()
average_duration = timereps(100, lambda : up(tensor_img))
print(f'nn.Upsample (CPU): {average_duration}')

average_duration = timereps(100, lambda : upsample(tensor_img, scale_factor=2, mode='bilinear', align_corners=False))
print(f'nn.functional.upsample (CPU) : {average_duration}')

img = Image.open('data/britney.png')
average_duration = timereps(100, lambda : resize(img, [img.size[0]*2, img.size[1]*2]))
print(f'torchvision resize : {average_duration}')
