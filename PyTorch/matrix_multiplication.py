import torch
import numpy as np

A_shape = [500, 5000]
B_shape = [5000, 500]
n_iter = 10

from time import time

def timereps(reps, func):
    start = time()
    for i in range(0, reps):
        func()
    end = time()
    return (end - start) / reps

assert A_shape[-1] == B_shape[0]
A = np.random.rand(*A_shape)
B = np.random.rand(*B_shape)

duration = timereps(n_iter, lambda:A@B)
print(f'np @ : {duration}')


A = torch.rand(A_shape)
B = torch.rand(B_shape)

duration = timereps(n_iter, lambda:A@B)
print(f'torch CPU @ : {duration}')

A = torch.rand(A_shape).cuda()
B = torch.rand(B_shape).cuda()

duration = timereps(n_iter, lambda:A@B)
print(f'torch GPU @ : {duration}')


