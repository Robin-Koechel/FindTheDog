import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

class compareImages():
    picOne = ""
    picTwo = ""

    model = models.resnet18(pretrained=True)
    layer = model._modules.get('avgpool')
    model.eval()

    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    toTensor = transforms.ToTensor()

    def __init__(self, img1Path, img2Path):
        self.picOne = img1Path
        self.picTwo = img2Path

    def getVector(self,image_name):
        img = Image.open(image_name)
        imgTemp = Variable(self.normalize(self.toTensor(self.scaler(img))).unsqueeze(0))

        my_embedding = torch.zeros(512)
        def copyData(m, i, o):
            my_embedding.copy_(o.data.reshape(o.data.size(1)))
        h = self.layer.register_forward_hook(copyData)
        self.model(imgTemp)
        h.remove()
        return my_embedding

    def compareVectors(self):
        vectorImgOne = self.getVector(self.picOne)
        vectorImgTwo = self.getVector(self.picTwo)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(vectorImgOne.unsqueeze(0), vectorImgTwo.unsqueeze(0))
        return cos_sim

#comp = compareImages()
#print('\nImages match to '+str(round(float(comp.compareVectors()*100),2)).strip("tensor([").strip("])")+"% überein")
