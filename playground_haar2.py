import os
import cv2

# see  https://github.com/opencv/opencv/tree/master/data/haarcascades

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
hm_frontal = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
hm_eye = os.path.join(cv2_base_dir, 'data/haarcascade_eye.xml')
hm_cat = os.path.join(cv2_base_dir, 'data/haarcascade_frontalcatface_extended.xml')
# hm_smile = os.path.join(cv2_base_dir, 'data/haarcascade_smile.xml')


face_cascade = cv2.CascadeClassifier(hm_frontal)
eye_cascade = cv2.CascadeClassifier(hm_eye)
cat_cascade = cv2.CascadeClassifier(hm_cat)

# img = cv2.imread("./images/TestImages/guy.jpg")
img = cv2.imread("./images/TestImages/group.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(faces)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray # gray[y:y + h, x:x + w]
    roi_color = img # img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


cv2.imshow("result image", roi_color)
cv2.waitKey(0)

cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cats = cat_cascade.detectMultiScale(roi_gray)
    for (cx, cy, cw, ch) in cats:
        cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), (0, 255, 255), 2)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
