import cv2
import sys

imagePath = sys.argv[1]

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

print("[INFO] Mendapatkan {0} Wajah!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)
    cv_gray = gray[y:y+h, x:x+w]
    cv_color = image[y:y+h, x:x+w]

    eyes = eye.detectMultiScale(cv_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(cv_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 5)

status = cv2.imwrite('images/hasil.jpg', image)
print("[INFO] Image hasil.jpg berhasil tersimpan: ", status)