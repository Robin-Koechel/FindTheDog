import ImageProcessing.ClassifyPretrained.ClassifyPre
import ImageProcessing.SegmentationPretrained.fcnResNet
from ImageProcessing.ColorRecognition import ColorRecognition

import matplotlib.pyplot as plt
import cv2

img_path = "F:\PYTHON\Projekte\FindTheDog\ImageProcessing\TestImages\cat.jpg"
img_raw = cv2.imread(img_path)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
plt.imshow(img_raw)
plt.show()

segmi = ImageProcessing.SegmentationPretrained.fcnResNet.SegmentationPretrained(img_path)
img_box, pet, boxLst, lstIndex = segmi.instance_segmentation()
print("Pet: ", pet)
plt.show()
crop_img = img_raw[int(boxLst[lstIndex][0][1]):int(boxLst[lstIndex][1][1]), int(boxLst[lstIndex][0][0]):int(boxLst[lstIndex][1][0])]
plt.imshow(crop_img)
plt.show()

cv2.imwrite(r'Crop.png', cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
classi = ImageProcessing.ClassifyPretrained.ClassifyPre.PretrainedClassifier("F:\PYTHON\Projekte\FindTheDog\Crop.png")
classi.classify()
classi.finalRes()
print("\n")

col = ColorRecognition.ColourInImage()
print("Dominant"+str(col.closestColor(col.getDominantColour())) + " ==> " + str(col.getDominantColour()))
print("Average"+ str(col.closestColor(col.getAvgColour())) + " ==> " + str(col.getAvgColour()))
exit(0)