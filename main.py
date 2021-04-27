import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir


normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],   #every number stands for a color channel RGB
    std=[0.229, 0.224, 0.225]       #maybe change Nubers if necessary
)


transform = transforms.Compose([
    transforms.Resize(256),         #make all imgs the same size
    transforms.CenterCrop(256),     #cut in direction center
    transforms.ToTensor(),          #transform img to tensor
    normalize])

train_data_list=[]
train_data = []
target_list = []

for i in listdir("images/Images/"):
    for j in listdir("images/Images/"+i+"/"):

        img = Image.open("images/Images/"+i+"/"+j)     #load example image from images
        img_tensor = transform(img)         #matrix now (3, 256,256) AND uses transformations and normalisations
        train_data_list.append(img_tensor)

        #***********************************************************
        #insert breeds
        #***********************************************************

        target_list.append(breeds)
        if len(train_data_list) >= 64:     #sets batchsize
            train_data.append((torch.stack(train_data_list),target_list))
            train_data_list = []        #clears list

print(train_data)

