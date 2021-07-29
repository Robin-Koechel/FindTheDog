class PretrainedClassifier:
    from torchvision import models
    import torch
    from torchvision import transforms
    from PIL import Image
    breedLst = []
    percentageLst = []
    sets = []
    img = ""

    def __init__(self,image):

        dir(self.models)
        self.sets = [self.models.alexnet(pretrained=True),self.models.densenet201(pretrained=True),self.models.googlenet(pretrained=True),
                self.models.inception_v3(pretrained=True),self.models.mobilenet_v2(pretrained=True),self.models.mobilenet_v3_large(pretrained=True)]

        self.img = self.Image.open(image)

    def classify(self):
        for set in self.sets:
            self.img.resize((256,256))

            transform = self.transforms.Compose([
                self.transforms.Resize(256),
                self.transforms.CenterCrop(224),
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])

            img_t = transform(self.img)
            batch_t = self.torch.unsqueeze(img_t,0)
            set.eval()

            out = set(batch_t)

            with open('F:\PYTHON\Projekte\FindTheDog\ClassifyPretrained\imagenet_classes.txt') as f:
                classes = [line.strip() for line in f.readlines()]
            #print("Number of classes: {}".format(len(classes)))

            _, indices = self.torch.sort(out, descending=True)
            percentage = self.torch.nn.functional.softmax(out, dim=1)[0] * 100
            #print([(classes[idx], percentage[idx].item()) for idx in indices[0][:3]])
            self.breedLst.append([classes[idx] for idx in indices[0][:1]])
            self.percentageLst.append([percentage[idx].item() for idx in indices[0][:1]])


    def finalRes(self):
        n = len(self.percentageLst)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                if self.percentageLst[j] > self.percentageLst[j + 1]:
                    self.percentageLst[j], self.percentageLst[j + 1] = self.percentageLst[j + 1], self.percentageLst[j]
                    self.breedLst[j], self.breedLst[j + 1] = self.breedLst[j + 1], self.breedLst[j]

        maxBreed = str(self.breedLst[len(self.breedLst)-1])
        minBreed = str(self.breedLst[0])

        maxPercentage = str(self.percentageLst[len(self.percentageLst)-1])
        minPercentage = str(self.percentageLst[0])

        print("BESTES ERGEBNIS: "+ maxBreed, maxPercentage)
        print("ZWEITBESTES ERGEBNIS: "+ str(self.breedLst[len(self.breedLst)-2]), str(self.percentageLst[len(self.percentageLst)-2]))
        print("Schlechtestes ERGEBNIS: " + minBreed, minPercentage)

    def bestRes(self):
        n = len(self.percentageLst)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                if self.percentageLst[j] > self.percentageLst[j + 1]:
                    self.percentageLst[j], self.percentageLst[j + 1] = self.percentageLst[j + 1], self.percentageLst[j]
                    self.breedLst[j], self.breedLst[j + 1] = self.breedLst[j + 1], self.breedLst[j]

        maxBreed = str(self.breedLst[len(self.breedLst) - 1])

        maxPercentage = str(self.percentageLst[len(self.percentageLst) - 1])

        print("BESTES ERGEBNIS: " + maxBreed, maxPercentage)


#classi = PretrainedClassifier("img/forestCat2.jpg")
#classi.classify()
#classi.finalRes()
