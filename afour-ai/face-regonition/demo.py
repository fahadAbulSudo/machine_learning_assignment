import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

 
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(f"Student names : {classNames}")
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%d/%m/%y %H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            print(f"\n{name} is marked present!")
 
testPath = 'Test-images'
myTestList = os.listdir(testPath)

for image in myTestList:
    curImg = cv2.imread(f'{testPath}/{image}')
    imgS = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)[0]
    
    matches = face_recognition.compare_faces(encodeListKnown,encodesCurFrame)
    faceDis = face_recognition.face_distance(encodeListKnown,encodesCurFrame)

    matchIndex = np.argmin(faceDis)
    window_title = "test-image"
    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        y1,x2,y2,x1 = facesCurFrame[0]
        cv2.rectangle(curImg,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(curImg,(x1,y2+35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(curImg,name,(x1,y2+20),cv2.FONT_HERSHEY_COMPLEX,0.35,(255,255,255),1)
        markAttendance(name)
    else:
        window_title = "Face not among registered faces!"
    
             
    cv2.imshow(window_title, curImg)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 






# Code to enable image detection via webcam
# cap = cv2.VideoCapture(0)


# while True:
#     success, img = cap.read()
#     imgS = cv2.resize(img,(0,0),None,0.25,0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
#     for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             y1,x2,y2,x1 = faceLoc
#             y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
#             markAttendance(name)
            
#     cv2.imshow('Webcam',img)
#     ## 27 is ascii value for ESC key
#     if cv2.waitKey(1) == 27:
#         break
    

# cap.release()
# cv2.destroyAllWindows()
