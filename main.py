import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import random

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # every number stands for a color channel RGB
    std=[0.229, 0.224, 0.225]  # maybe change Nubers if necessary
)

transform = transforms.Compose([
    transforms.Resize(256),  # make all imgs the same size
    transforms.CenterCrop(256),  # cut in direction center
    transforms.ToTensor(),  # transform img to tensor
    normalize])

train_data_list = []
train_data = []
target_list = []

files = listdir("images/AllDogImages/")     #creates list with all imges of breed i
for n in range(len(listdir("images/AllDogImages/"))):
    # random to avoid learn by heart
    j = random.choice(files)            #chooses random image from list files
    files.remove(j)                     #removes that file to avoid double data

    img = Image.open("images/AllDogImages/" + j)  # load example image from images
    img_tensor = transform(img)  # matrix now (3, 256,256) AND uses transformations and normalisations
    train_data_list.append(img_tensor)

    # ***********************************************************
    # insert breeds
    # ***********************************************************
    # binary classification nessecary if softmax algorithm should be used
    dogBreed1 = 1 if 'Chihuahua' in j else 0
    dogBreed2 = 1 if 'Japanese_spaniel' in j else 0
    # and so on ...

    target = [dogBreed1, dogBreed2]

    target_list.append(target)
    if len(train_data_list) >= 64:  # sets batchsize
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []  # clears list
        break
print(train_data)

