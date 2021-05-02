import cv2
import random
import os

image_path = "D:\\pycharm\\LearnersLearners\\PetImages\\Cat"




cascade_classifier = "./models/haarcascades/haarcascade_frontalcatface_extended.xml"
# cascade_classifier = "./models/haarcascades/haarcascade_frontalcatface.xml"


def populate_test_images(i_path):
    t_imgs = []
    for directory, sub_dir_list, file_list in os.walk(i_path):
        if len(file_list) > 0:
            file_name_and_path = directory + "\\" + file_list[random.randint(0, len(file_list) - 1)]

            if file_name_and_path.lower().endswith((
                    '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                t_imgs.append(file_name_and_path)
    return t_imgs


test_images = populate_test_images(image_path)
rects = None
detector = cv2.CascadeClassifier(cascade_classifier)

for img in test_images:
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#    rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))
    rects = detector.detectMultiScale(gray, 1.3, 5)
    print(img, rects)

    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imshow("Cat Faces", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
