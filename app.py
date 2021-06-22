import cv2

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')

while True:
    isTrue, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv_gray = gray[y:y+h, x:x+w]
        cv_color = img[y:y+h, x:x+w]

        eyes = eye.detectMultiScale(cv_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(cv_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 5)

    cv2.imshow('Video', img)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()