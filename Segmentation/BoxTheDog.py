import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.patches as patches

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_Instance_Category_Names =[  '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',  'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


print("Anzahl Klassen: ",len(COCO_Instance_Category_Names))

def get_prediction(img_path,threshold=0.5):
    img = Image.open(img_path)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)

    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_class = [COCO_Instance_Category_Names[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class


def instance_segmentation(img_path, threshold=0.5):
    boxes, pred_cls = get_prediction(img_path, threshold=threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    counter  = 0
    petLst= ['bird', 'cat', 'dog', 'horse', 'sheep','cow','elephant', 'bear', 'zebra', 'giraffe']
    for i in pred_cls:
        if i not in petLst:
            counter+=1
        else:
            print(pred_cls)
            fig, ax = plt.subplots()
            rect = patches.Rectangle((boxes[counter][0][0], boxes[counter][0][1]), boxes[counter][1][0] - boxes[counter][0][0], boxes[counter][1][1] - boxes[counter][0][1], linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.imshow(img)
            break
    return img, pred_cls

img, pred_classes = instance_segmentation("F:\PYTHON\Projekte\FindTheDog\Pretrained\img\mensch-vs-hund.jpg")
plt.show()

fig, ax = plt.subplots() #required -- box

plt.imshow(I);plt.axis('off')

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds) # meta data


#stuff needed for Box
rect = patches.Rectangle([anns[0]['bbox'][0], anns[0]['bbox'][1]] , anns[0]['bbox'][2], anns[0]['bbox'][3], linewidth=3, edgecolor='r', facecolor='none')
ax.add_patch(rect)

coco.showAnns(anns) #shows colored body
plt.show()
