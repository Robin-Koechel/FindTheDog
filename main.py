import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

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

    img = Image.open("images/AllDogImages/" + j).convert('RGB')  # load example image from images
    img_tensor = transform(img)  # matrix now (3, 256,256) AND uses transformations and normalisations

    train_data_list.append(img_tensor)

    #OUTPUT
    def breedClassification():
        affenpinscher = 1 if 'affenpinscher' in j else 0
        Afghan_hound = 1 if 'Afghan_hound' in j else 0
        African_hunting_dog = 1 if 'African_hunting_dog' in j else 0
        Airedale = 1 if 'Airedale' in j else 0
        American_Staffordshire_terrier = 1 if 'American_Staffordshire_terrier' in j else 0
        Appenzeller = 1 if 'Appenzeller' in j else 0
        Australian_terrier = 1 if 'Australian_terrier' in j else 0
        basenji = 1 if 'basenji' in j else 0
        basset = 1 if 'basset' in j else 0
        beagle = 1 if 'beagle' in j else 0
        Bedlington_terrier = 1 if 'Bedlington_terrier' in j else 0
        Bernese_mountain_dog = 1 if 'Bernese_mountain_dog' in j else 0
        black_and_tan_coonhound = 1 if 'black-and-tan_coonhound' in j else 0
        Blenheim_spaniel = 1 if 'Blenheim_spaniel' in j else 0
        bloodhound = 1 if 'bloodhound' in j else 0
        bluetick = 1 if 'bluetick' in j else 0
        Border_collie = 1 if 'Border_collie' in j else 0
        Border_terrier = 1 if 'Border_terrier' in j else 0
        borzoi = 1 if 'borzoi' in j else 0
        Boston_bull = 1 if 'Boston_bull' in j else 0
        Bouvier_des_Flandres = 1 if 'Bouvier_des_Flandres' in j else 0
        boxer = 1 if 'boxer' in j else 0
        Brabancon_griffon = 1 if 'Brabancon_griffon' in j else 0
        briard = 1 if 'briard' in j else 0
        Brittany_spaniel = 1 if 'Brittany_spaniel' in j else 0
        bull_mastiff = 1 if 'bull_mastiff' in j else 0
        cairn = 1 if 'cairn' in j else 0
        Cardigan = 1 if 'Cardigan' in j else 0
        Chesapeake_Bay_retriever = 1 if 'Chesapeake_Bay_retriever' in j else 0
        Chihuahua = 1 if 'Chihuahua' in j else 0
        chow = 1 if 'chow' in j else 0
        clumber = 1 if 'clumber' in j else 0
        cocker_spaniel = 1 if 'cocker_spaniel' in j else 0
        collie = 1 if 'collie' in j else 0
        curly_coated_retriever = 1 if 'curly-coated_retriever' in j else 0
        Dandie_Dinmont = 1 if 'Dandie_Dinmont' in j else 0
        dhole = 1 if 'dhole' in j else 0
        dingo = 1 if 'dingo' in j else 0
        Doberman = 1 if 'Doberman' in j else 0
        English_foxhound = 1 if 'English_foxhound' in j else 0
        English_setter = 1 if 'English_setter' in j else 0
        English_springer = 1 if 'English_springer' in j else 0
        EntleBucher = 1 if 'EntleBucher' in j else 0
        Eskimo_dog = 1 if 'Eskimo_dog' in j else 0
        flat_coated_retriever = 1 if 'flat-coated_retriever' in j else 0
        French_bulldog = 1 if 'French_bulldog' in j else 0
        German_shepherd = 1 if 'German_shepherd' in j else 0
        German_short_haired_pointer = 1 if 'German_short-haired_pointer' in j else 0
        giant_schnauzer = 1 if 'giant_schnauzer' in j else 0
        golden_retriever = 1 if 'golden_retriever' in j else 0
        Gordon_setter = 1 if 'Gordon_setter' in j else 0
        Greater_Swiss_Mountain_dog = 1 if 'Greater_Swiss_Mountain_dog' in j else 0
        Great_Dane = 1 if 'Great_Dane' in j else 0
        Great_Pyrenees = 1 if 'Great_Pyrenees' in j else 0
        groenendael = 1 if 'groenendael' in j else 0
        Ibizan_hound = 1 if 'Ibizan_hound' in j else 0
        Irish_setter = 1 if 'Irish_setter' in j else 0
        Irish_terrier = 1 if 'Irish_terrier' in j else 0
        Irish_water_spaniel = 1 if 'Irish_water_spaniel' in j else 0
        Irish_wolfhound = 1 if 'Irish_wolfhound' in j else 0
        Italian_greyhound = 1 if 'Italian_greyhound' in j else 0
        Japanese_spaniel = 1 if 'Japanese_spaniel' in j else 0
        keeshond = 1 if 'keeshond' in j else 0
        kelpie = 1 if 'kelpie' in j else 0
        Kerry_blue_terrier = 1 if 'Kerry_blue_terrier' in j else 0
        komondor = 1 if 'komondor' in j else 0
        kuvasz = 1 if 'kuvasz' in j else 0
        Labrador_retriever = 1 if 'Labrador_retriever' in j else 0
        Lakeland_terrier = 1 if 'Lakeland_terrier' in j else 0
        Leonberg = 1 if 'Leonberg' in j else 0
        Lhasa = 1 if 'Lhasa' in j else 0
        malamute = 1 if 'malamute' in j else 0
        malinois = 1 if 'malinois' in j else 0
        Maltese_dog = 1 if 'Maltese_dog' in j else 0
        Mexican_hairless = 1 if 'Mexican_hairless' in j else 0
        miniature_pinscher = 1 if 'miniature_pinscher' in j else 0
        miniature_poodle = 1 if 'miniature_poodle' in j else 0
        miniature_schnauzer = 1 if 'miniature_schnauzer' in j else 0
        Newfoundland = 1 if 'Newfoundland' in j else 0
        Norfolk_terrier = 1 if 'Norfolk_terrier' in j else 0
        Norwegian_elkhound = 1 if 'Norwegian_elkhound' in j else 0
        Norwich_terrier = 1 if 'Norwich_terrier' in j else 0
        Old_English_sheepdog = 1 if 'Old_English_sheepdog' in j else 0
        otterhound = 1 if 'otterhound' in j else 0
        papillon = 1 if 'papillon' in j else 0
        Pekinese = 1 if 'Pekinese' in j else 0
        Pembroke = 1 if 'Pembroke' in j else 0
        Pomeranian = 1 if 'Pomeranian' in j else 0
        pug = 1 if 'pug' in j else 0
        redbone = 1 if 'redbone' in j else 0
        Rhodesian_ridgeback = 1 if 'Rhodesian_ridgeback' in j else 0
        Rottweiler = 1 if 'Rottweiler' in j else 0
        Saint_Bernard = 1 if 'Saint_Bernard' in j else 0
        Saluki = 1 if 'Saluki' in j else 0
        Samoyed = 1 if 'Samoyed' in j else 0
        schipperke = 1 if 'schipperke' in j else 0
        Scotch_terrier = 1 if 'Scotch_terrier' in j else 0
        Scottish_deerhound = 1 if 'Scottish_deerhound' in j else 0
        Sealyham_terrier = 1 if 'Sealyham_terrier' in j else 0
        Shetland_sheepdog = 1 if 'Shetland_sheepdog' in j else 0
        Shih_Tzu = 1 if 'Shih-Tzu' in j else 0
        Siberian_husky = 1 if 'Siberian_husky' in j else 0
        silky_terrier = 1 if 'silky_terrier' in j else 0
        soft_coated_wheaten_terrier = 1 if 'soft-coated_wheaten_terrier' in j else 0
        Staffordshire_bullterrier = 1 if 'Staffordshire_bullterrier' in j else 0
        standard_poodle = 1 if 'standard_poodle' in j else 0
        standard_schnauzer = 1 if 'standard_schnauzer' in j else 0
        Sussex_spaniel = 1 if 'Sussex_spaniel' in j else 0
        Tibetan_mastiff = 1 if 'Tibetan_mastiff' in j else 0
        Tibetan_terrier = 1 if 'Tibetan_terrier' in j else 0
        toy_poodle = 1 if 'toy_poodle' in j else 0
        toy_terrier = 1 if 'toy_terrier' in j else 0
        vizsla = 1 if 'vizsla' in j else 0
        Walker_hound = 1 if 'Walker_hound' in j else 0
        Weimaraner = 1 if 'Weimaraner' in j else 0
        Welsh_springer_spaniel = 1 if 'Welsh_springer_spaniel' in j else 0
        West_Highland_white_terrier = 1 if 'West_Highland_white_terrier' in j else 0
        whippet = 1 if 'whippet' in j else 0
        wire_haired_fox_terrier = 1 if 'wire-haired_fox_terrier' in j else 0
        Yorkshire_terrier = 1 if 'Yorkshire_terrier' in j else 0

        target = [affenpinscher, Afghan_hound, African_hunting_dog, Airedale, American_Staffordshire_terrier,
                  Appenzeller,
                  Australian_terrier, basenji, basset, beagle, Bedlington_terrier, Bernese_mountain_dog,
                  black_and_tan_coonhound,
                  Blenheim_spaniel, bloodhound, bluetick, Border_collie, Border_terrier, borzoi, Boston_bull,
                  Bouvier_des_Flandres,
                  boxer, Brabancon_griffon, briard, Brittany_spaniel, bull_mastiff, cairn, Cardigan,
                  Chesapeake_Bay_retriever,
                  Chihuahua, chow, clumber, cocker_spaniel, collie, curly_coated_retriever, Dandie_Dinmont, dhole,
                  dingo, Doberman,
                  English_foxhound, English_setter, English_springer, EntleBucher, Eskimo_dog, flat_coated_retriever,
                  French_bulldog, German_shepherd, German_short_haired_pointer, giant_schnauzer, golden_retriever,
                  Gordon_setter,
                  Greater_Swiss_Mountain_dog, Great_Dane, Great_Pyrenees, groenendael, Ibizan_hound, Irish_setter,
                  Irish_terrier,
                  Irish_water_spaniel, Irish_wolfhound, Italian_greyhound, Japanese_spaniel, keeshond, kelpie,
                  Kerry_blue_terrier,
                  komondor, kuvasz, Labrador_retriever, Lakeland_terrier, Leonberg, Lhasa, malamute, malinois,
                  Maltese_dog,
                  Mexican_hairless, miniature_pinscher, miniature_poodle, miniature_schnauzer, Newfoundland,
                  Norfolk_terrier,
                  Norwegian_elkhound, Norwich_terrier, Old_English_sheepdog, otterhound, papillon, Pekinese, Pembroke,
                  Pomeranian,
                  pug, redbone, Rhodesian_ridgeback, Rottweiler, Saint_Bernard, Saluki, Samoyed, schipperke,
                  Scotch_terrier,
                  Scottish_deerhound, Sealyham_terrier, Shetland_sheepdog, Shih_Tzu, Siberian_husky, silky_terrier,
                  soft_coated_wheaten_terrier, Staffordshire_bullterrier, standard_poodle, standard_schnauzer,
                  Sussex_spaniel,
                  Tibetan_mastiff, Tibetan_terrier, toy_poodle, toy_terrier, vizsla, Walker_hound, Weimaraner,
                  Welsh_springer_spaniel, West_Highland_white_terrier, whippet, wire_haired_fox_terrier,
                  Yorkshire_terrier, ]
        return target
    breedClassification()

    target_list.append(breedClassification())
    if len(train_data_list) >= 64:  # sets batchsize
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []  # clears list

        print('Loaded batch ', len(train_data), ' of ', int(len(listdir('images/AllDogImages/'))/64))
        percentage = (len(train_data)/int(len(listdir('images/AllDogImages/'))/64))*100
        print('Percentage done', str(percentage)[0:4], '%')

#print(train_data)



class Netz(nn.Module):
    def __init__(self):     #Constructor
        super(Netz,self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size = 5)       #convolutional layer 1
        self.conv2 = nn.Conv2d(5,10, kernel_size=5)         #convolutional layer 2
        self.conv3 = nn.Conv2d(10,20, kernel_size=5)        #convolutional layer 3
        self.fc1 = nn.Linear(15680,1000)                    #fully connected layer 1 15680--> Neurons
        self.fc2 = nn.Linear(1000, 119)                       #fully connected layer 2 2 --> Output (Change)


    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(x,119)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,119)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 119)
        x = F.relu(x)
        x = x.view(-1, 15680)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x)


model = Netz()


optimizer = optim.Adam(model.parameters(), lr=0.01)
def train(epoch):
    model.train()
    batch_id=0
    for data, target in train_data:
        target = torch.Tensor(target)
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('loss ' + loss)
        batch_id += 1

for epoch in range(1,30):
    train(epoch)
