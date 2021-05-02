import abc
import os
import random

import cv2
import numpy as np

# some setup
img_path = "images"


class Approach:
    __metaclass__ = abc.ABCMeta

    def __init__(self, image_path):
        self.test_images = []
        self.image_path = image_path
        self.populate_test_images()
        self.class_names = []
        self.image_net_names = []
        self.model = None
        self.COLORS = None

    def populate_test_images(self):
        for directory, sub_dir_list, file_list in os.walk(self.image_path):
            if len(file_list) > 0:
                file_name_and_path = directory + "\\" + file_list[random.randint(0, len(file_list)-1)]

                if file_name_and_path.lower().endswith((
                        '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    self.test_images.append(file_name_and_path)

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
        self.model = cv2.dnn.readNet(model="models/trained/frozen_inference_graph.pb",
                                     config="models/trained/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt",
                                     framework="TensorFlow")
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
                confidence = detection[2]
                if confidence > .4:
                    class_id = detection[1]
                    class_name = self.class_names[int(class_id)-1]
                    color = self.COLORS[int(class_id)]
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    cv2.rectangle(image, (int(box_x), int(box_y)),
                                  (int(box_width), int(box_height)), color, thickness=1)
                    cv2.putText(image, class_name, (int(box_x), int(box_y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    cv2.imshow('image', image)
                    # cv2.imwrite('./outputs/image_result.jpg', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


class ImageNetClassification(Approach):
    def initialize(self):
        with open('models/trained/classification_classes_ILSVRC2012.txt', 'r') as f:
            self.image_net_names = f.read().split('\n')
        self.class_names = [name.split(',')[0] for name in self.image_net_names]
        self.model = cv2.dnn.readNetFromCaffe("models/trained/DenseNet_121.prototxt",
                                              "models/trained/DenseNet_121.caffemodel")

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
            probabilities = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
            final_prob = np.max(probabilities) * 100.
            out_name = self.class_names[label_id]
            out_text = f"{out_name}, {final_prob:.3f}"
            cv2.putText(image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


coco = CocoDetection(img_path)
coco.initialize()
coco.perform()

inc = ImageNetClassification(img_path)
inc.initialize()
inc.perform()
