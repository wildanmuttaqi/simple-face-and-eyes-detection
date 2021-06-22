import cv2

image = cv2.imread('einstein.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv_warna = image[y:y+h, x:x+w]
    cv_gray = gray[y:y+h, x:x+w]

cv2.imwrite('image/hasil.jpg', image)
cv2.destroyAllWindows()