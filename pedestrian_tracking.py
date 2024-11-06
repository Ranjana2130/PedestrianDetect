import cv2

full_cascade=cv2.CascadeClassifier("C:/Users/akash.patwa/Downloads/haarcascade_fullbody.xml")

vid=cv2.VideoCapture("C:/Users/akash.patwa/Downloads/walking.avi")

while(True):
    ret,frame=vid.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    body=full_cascade.detectMultiScale(gray,1.1,8)
    for x,y,w,h in body:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,150,255),2)
    cv2.imshow("Fullbody Detection", frame)
    if cv2.waitKey(25)==32:
        break
vid.release()
cv2.destroyAllWindows
