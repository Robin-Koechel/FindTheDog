import ImageProcessing.ClassifyPretrained.ClassifyPre
import ImageProcessing.SegmentationPretrained.fcnResNet
from ImageProcessing.ColorRecognition import ColorRecognition
from ImageProcessing.CompareImages import compare
from Scraping import FindeFixScraper
import cv2


imgPath = "F:\PYTHON\Projekte\FindTheDog\ImageProcessing\TestImages\cat.jpg"
imgCropPath = "F:\PYTHON\Projekte\FindTheDog\Crop.png";


findeFix = FindeFixScraper.FindeFix()
findeFix.scrapePets(1)
lstPets = findeFix.getPetLst()

def processInputImage(imgPath):
    cropImage(imgPath)

def processAll(lstPets):
    for pet in lstPets:
        try:
            cropImage(pet.getImagePath())
            compareImages(pet.getImagePath())
            print(pet.outAll())
        except:
            print("Fehler bei einem Bild")

def cropImage(imgPath="Crop.png"):
    imgRaw = cv2.imread(imgPath)
    imgRaw = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2RGB)
    segmi = ImageProcessing.SegmentationPretrained.fcnResNet.SegmentationPretrained(imgPath)
    imgBox, pet, boxLst, lstIndex = segmi.instance_segmentation()
    #print("Pet: ", pet)
    cropImg = imgRaw[int(boxLst[lstIndex][0][1]):int(boxLst[lstIndex][1][1]), int(boxLst[lstIndex][0][0]):int(boxLst[lstIndex][1][0])]
    cv2.imwrite(imgPath, cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB))

    classifyImage()

def classifyImage():
    classi = ImageProcessing.ClassifyPretrained.ClassifyPre.PretrainedClassifier(imgCropPath)
    classi.classify()
    classi.finalRes()


def getColourStuff():
    col = ColorRecognition.ColourInImage()
    print("Dominant"+str(col.closestColor(col.getDominantColour())) + " ==> " + str(col.getDominantColour()))
    print("Average"+ str(col.closestColor(col.getAvgColour())) + " ==> " + str(col.getAvgColour()))

def compareImages(imgPath):
    compareImages = compare.compareImages("Crop.png",imgPath)
    print('\nImages match to '+str(round(float(compareImages.compareVectors()*100),2)).strip("tensor([").strip("])")+"%")

processInputImage(imgPath)
processAll(lstPets)
