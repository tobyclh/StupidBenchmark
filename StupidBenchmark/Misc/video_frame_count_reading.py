import torch
import numpy as np
import imageio
import cv2

from time import time
def timereps(reps, func):
    start = time()
    for i in range(0, reps):
        func()
    end = time()
    return (end - start) / reps

path_to_video = '/media/toby/Blade/Users/tobyc/Documents/Voice2Face/data/GRID/s2/swwc5p.mpg'
n_iter = 100

def get_video_length_imageio():
    video = imageio.get_reader(path_to_video, 'ffmpeg')
    return video._meta['nframes']

def get_video_length_cv2():
    cap = cv2.VideoCapture(str(path_to_video))
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = timereps(n_iter, get_video_length_imageio)
print(f'image io get length : {duration}')

duration = timereps(n_iter, get_video_length_cv2)
print(f'cv2 get length : {duration}')



