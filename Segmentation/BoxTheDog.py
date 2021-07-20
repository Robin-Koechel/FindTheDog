from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

matplotlib.use('TkAgg')
dataDir = './COCOdataset2017'
dataType='val'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

filteredClasses = ['cat']
catIds = coco.getCatIds(catNms=filteredClasses)
imgIds = coco.getImgIds(catIds=catIds)


#img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
img = coco.loadImgs(imgIds[10])[0] #SELECTS THE IMAGE

I = io.imread('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))/255.0 #i think it gets the meta data

#plt.axis('off')
#plt.imshow(I)
#plt.show()


fig, ax = plt.subplots() #required -- box

plt.imshow(I);plt.axis('off')

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds) # meta data


#stuff needed for Box
rect = patches.Rectangle([anns[0]['bbox'][0], anns[0]['bbox'][1]] , anns[0]['bbox'][2], anns[0]['bbox'][3], linewidth=3, edgecolor='r', facecolor='none')
ax.add_patch(rect)

coco.showAnns(anns) #shows colored body
plt.show()
