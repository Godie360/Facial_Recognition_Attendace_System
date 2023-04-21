import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'image_sample'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEcondings(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open("ATTENDANCE.CSV",'r+')as f:
        myDataList = f.readlines()
        nameList = []
        for line in  myDataList:
            entry = line.split(' ,')
            nameList.append(entry[0])
        if name not in nameList:
            nowT = datetime.now().date()
            nowD = datetime.now().time()

            Timestring = nowT .strftime('%H:%M:%S')
            Datestring = nowD.strftime('%DD/%MM/%YYYY')
            f.writelines(f'\n{name},{Timestring}')
            f.writelines(f'\n{name},{Datestring}')
markAttendance('juma')

encodeListknown = findEcondings(images)
print("ENCODING COMPLETE SUCCESSFULLY")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)[0]
    encodeCurFrame = face_recognition.face_encodings(imgs,face_recognition)[0]

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)


  #  cv2.imshow("webcam",img)
    cv2.waitKey(1)
    y1,x2,y2,x1 = faceLoc
    y1, x2, y2, x1 =  y1*4,x2*4,y2*4,x1*4
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.rectangle(img,(x1,y1-35),(x2,y2),(x2,y2),(0,255,0),cv2.FILLED)
    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,225,225),2)
    markAttendance(name)










