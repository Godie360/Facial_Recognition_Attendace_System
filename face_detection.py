import cv2
import numpy as np
import face_recognition

#  function for resizing the images
def resize(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


christina = face_recognition.load_image_file('image_sample/tina.jpg')
christina = resize(christina, 0.30)
christina = cv2.cvtColor(christina, cv2.COLOR_BGR2RGB)

christina_test = face_recognition.load_image_file('image_sample/tina3.jpg')
christina_test = resize(christina_test, 1.30)
christina_test = cv2.cvtColor(christina_test, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(christina)[0]
encode_christina = face_recognition.face_encodings(christina)[0]
cv2.rectangle(christina,( faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]), (255,0,255),2)

faceloc_test = face_recognition.face_locations(christina_test)[0]
encode_christina_test = face_recognition.face_encodings(christina_test)[0]
cv2.rectangle(christina_test,( faceloc_test[3],faceloc_test[0]),(faceloc_test[1],faceloc_test[2]), (255,0,255),2)

result = face_recognition.compare_faces([encode_christina],encode_christina_test)
face_dis = face_recognition.face_distance([encode_christina], encode_christina_test)
print(result, face_dis)
cv2.putText(christina_test,f'{result}{round(face_dis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255),2)


cv2.imshow('tina', christina)
cv2.imshow('tina3', christina_test)
cv2.waitKey(0)


