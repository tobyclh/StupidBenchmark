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
video = imageio.get_reader(path_to_video, 'ffmpeg')
video_length_imageio = int(video._meta['nframes'])


cap = cv2.VideoCapture(str(path_to_video))
video_length_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

n_iter = 100

def video_frame_random_access_imageio():
    frame_idx = np.random.randint(0, video_length)
    video = imageio.get_reader(path_to_video, 'ffmpeg')
    return video.get_data(frame_idx)

def video_frame_random_access_cv2():
    frame_idx = np.random.randint(0, video_length)
    cap = cv2.VideoCapture(str(path_to_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
    ret, frame = cap.read()
    return frame[...,::-1]

duration = timereps(n_iter, video_frame_random_access_imageio)
print(f'image io get frame : {duration}')

duration = timereps(n_iter, video_frame_random_access_cv2)
print(f'cv2 get frame : {duration}')



