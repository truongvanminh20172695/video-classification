

import cv2
import os
import time

import utils


import seaborn as sn
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from PIL import Image
from sklearn.metrics import multilabel_confusion_matrix
import torchvision
import torch

import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# contruct the argument parser
# ap = argparse.ArgumentParser()
# ap.add_argument('-m', '--model',
#                 help='path to save the trained model')
# ap.add_argument('-e', '--epochs', type=int, default=1,
#                 help='number of epochs to train our network for')
# args = vars(ap.parse_args())


# check cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}')

# get clip and label
df = pd.read_csv('data_1_OF_test.csv')
clips = df['clip'].tolist()

# change string type to array type
def alter_type(clip):
    new_clip_array = clip.replace("'", "").replace("[", "").replace("]", "").replace("/content/drive/MyDrive/Colab Notebooks/frames_temp", "frames_OF_test").split(", ")
    return new_clip_array

# process all clip_len frames to get X
def get_clip_input(all_clips):
    X = []
    for each_clip in all_clips:
        each_clip = alter_type(each_clip)
        X.append(each_clip)
    return X

# get X, y
X = get_clip_input(clips)
y = df['label'].tolist()
print('length X: ', len(X))
print('length y: ', len(y))

# train, test split
# (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
# print(f'Training instances: {len(X_train)}')
# print(f'Validataion instances: {len(X_test)}')
# print('X-test',X_test)
# print('y_test',y_test)
# custom dataset
class UCF11(Dataset):
    def __init__(self, clips, labels):
        self.clips = clips
        self.labels = labels

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        clip = self.clips[i]
        input_frames = []
        for frame in clip:
            image = Image.open(frame)
            image = image.convert('RGB')
            image = np.array(image)
            image = utils.transform(image=image)['image']
            input_frames.append(image)
        input_frames = np.array(input_frames)
        # print('input_frames.shape: ', input_frames.shape)
        # input_frames = np.expand_dims(input_frames, axis=0)
        input_frames = np.transpose(input_frames, (3,0,1,2))
        input_frames = torch.tensor(input_frames, dtype=torch.float32)
        input_frames = input_frames.to(device)
        # label
        self.labels = np.array(self.labels)
        lb = LabelBinarizer()
        self.labels = lb.fit_transform(self.labels)
        label = self.labels[i]
        label = torch.tensor(label, dtype=torch.long)
        # print('label: ', label)
        label = label.to(device)
        return (input_frames, label)

# train_data = UCF11(X_train, y_train)
val_data = UCF11(X, y)
# print('val_data = ', val_data)
# print('val_data[0] = ',val_data[0])
# print('val_data[1] = ',val_data[1])
# learning params
lr = 1e-3
batch_size = 16

# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# model
# model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)
# for param in model.parameters():
#     param.requires_grad = False
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 5)
# model = model.to(device)
# print(model)
model = torchvision.models.video.r2plus1d_18(pretrained=False, progress=True)
for param in model.parameters():
    param.requires_grad = False
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 12)
model = model.to(device)
model.load_state_dict(torch.load('outputs/r2plus1d_TSN_OF_RGB.pth', map_location=torch.device('cpu')))



from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
y_pred = []
y_true = []
with torch.no_grad():
    for i, (data) in enumerate(val_loader):
        data, target = data[0].to(device), data[1].to(device)
        x = torch.max(target, 1)[1]
        y_true.extend(x)
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds)


# classes = ('tennis', 'spiking', 'walk')
classes = ('G1', 'G2', 'G3','G4','G5','G6','G7','G8','G9','G10','G11','G12')
cf_matrix = confusion_matrix(y_true , y_pred)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('outputs/confusion_matrix_r2plus1d_TSN_OF_1.png')


print('TRAINING COMPLETE')