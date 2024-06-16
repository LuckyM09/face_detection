import cv2

capture_face=cv2.CascadeClassifier("C:/Users/Asus/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalcatface.xml")
capture_vdo=cv2.VideoCapture(0)
while True:
    let, vdo=capture_vdo.read()
    color=cv2.cvtColor(vdo,cv2.COLOR_BGR2GRAY)
    faces=capture_face.detectMultiScale(
        color,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(vdo,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Live_vdo",vdo)
    if cv2.waitKey(10)==ord("a"):
        break

capture_vdo.release()    