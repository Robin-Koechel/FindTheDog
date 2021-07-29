import ClassifyPretrained.main
import SegmentationPretrained.fcnResNet
import matplotlib.pyplot as plt
import cv2

img_path = 'F:\PYTHON\Projekte\FindTheDog\ClassifyPretrained\img\hamster.jpg'
img_raw = cv2.imread(img_path)

plt.imshow(img_raw)
plt.show()

segmi = SegmentationPretrained.fcnResNet.SegmentationPretrained(img_path)
img_box, pet, boxLst,lstIndex = segmi.instance_segmentation()
print("Pet: ",pet)
plt.show()
crop_img = img_raw[int(boxLst[lstIndex][0][1]):int(boxLst[lstIndex][1][1]),int(boxLst[lstIndex][0][0]):int(boxLst[lstIndex][1][0])]
plt.imshow(crop_img)
plt.show()

cv2.imwrite(r'CropedImg\Crop.png',crop_img)
classi = ClassifyPretrained.main.PretrainedClassifier("CropedImg\Crop.png")
classi.classify()
classi.bestRes()
