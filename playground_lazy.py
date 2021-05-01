import cv2
import numpy as np
import os, random, abc

# some setup
img_path ="images"

class Approach():
    __metaclass__ = abc.ABCMeta

    def __init__(self, image_path):
        self.image_path = image_path
        self.populate_test_images()
        None

    def populate_test_images(self):
        self.test_images = []
        for directory, subdirlist, filelist in os.walk(self.image_path): 
            if len(filelist) > 0:
                self.test_images.append(directory + "\\" + filelist[random.randint(0,len(filelist)-1)])

    @abc.abstractmethod
    def initialize(self):
        """very foo documentation"""
        return

    @abc.abstractmethod
    def perform(self):
        """very bar documentation"""
        return

class CocoDetection(Approach):
    def initialize(self):
        with open('models/trained/object_detection_classes_coco.txt', 'r') as f:
            self.class_names = f.read().split('\n')
        self.model = cv2.dnn.readNet(model='models/trained/frozen_inference_graph.pb',
                                    config='models/trained/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                                    framework='TensorFlow')
        # get a different color array for each of the classes
        self.COLORS = np.random.uniform(0, 255, size=(len(self.class_names), 3))

    def perform(self):
        for i in self.test_images:
            print(i)
            image = cv2.imread(i)
            image_height, image_width, _ = image.shape
            blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                 swapRB=True)
            self.model.setInput(blob)
            output = self.model.forward()
            for detection in output[0, 0, :, :]:
                # extract the confidence of the detection
                confidence = detection[2]
                # draw bounding boxes only if the detection confidence is above...
                # ... a certain threshold, else skip
                if confidence > .4:
                # get the class id
                    class_id = detection[1]
                    # map the class id to the class
                    class_name = self.class_names[int(class_id)-1]
                    color = self.COLORS[int(class_id)]
                    # get the bounding box coordinates
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    # get the bounding box width and height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    # draw a rectangle around each detected object
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=1)
                    # put the FPS text on top of the frame
                    cv2.putText(image, class_name, (int(box_x), int(box_y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    cv2.imshow('image', image)
                    #cv2.imwrite('./outputs/image_result.jpg', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

class ImageNetClassification(Approach):
    def initialize(self):
        with open('models/trained/classification_classes_ILSVRC2012.txt', 'r') as f:
            self.image_net_names = f.read().split('\n')
        # final class names (just the first word of the many ImageNet names for one image)
        self.class_names = [name.split(',')[0] for name in self.image_net_names]
        self.model = cv2.dnn.readNetFromCaffe('models/trained/DenseNet_121.prototxt', 
                                            'models/trained/DenseNet_121.caffemodel')
    def perform(self):
        for i in self.test_images:
            print(i)
            image = cv2.imread(i)
            # Note: the mean values for the ImageNet training set are R=103.93, G=116.77, and B=123.68
            blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), 
                                         mean=(104, 117, 123), swapRB=False)
            self.model.setInput(blob)
            outputs = self.model.forward()

            final_outputs = outputs[0]
            final_outputs = final_outputs.reshape(1000, 1)
            label_id = np.argmax(final_outputs)
            probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
            final_prob = np.max(probs) * 100.
            out_name = self.class_names[label_id]
            out_text = f"{out_name}, {final_prob:.3f}"
            cv2.putText(image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

coco = CocoDetection(img_path)
coco.initialize()
coco.perform()

inc = ImageNetClassification(img_path)
inc.initialize()
inc.perform()
