import pandas as pd
import os
import numpy as np
import cv2
import glob
import shutil
from random import choice
from tqdm import tqdm
import subprocess


def stack_frames(frames, clip_len):
    frames_stacked = [frames[x:x+16] for x in range(0, len(frames)-15)][0::16]
    return frames_stacked

if __name__ == '__main__':

    data = {
        'clip' : [],
        'label' : []
    }
    # frames_dir = '/content/drive/MyDrive/R(2+1)D/test'
    # frames = glob.glob(f'{frames_dir}/*')
    frames = os.listdir('1')
    print(frames)
    frames_stacked = stack_frames(frames,16)
    data['clip'].extend(frame_stacked for frame_stacked in frames_stacked)
    data['label'].extend(frame_stacked[0].split('/')[-1].split('_')[1] for frame_stacked in frames_stacked)
    df = pd.DataFrame(data, columns=['clip', 'label'])
    df.to_csv('data_val_1.csv', index=False)