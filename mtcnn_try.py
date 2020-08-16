import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
# from keras import models
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

model = tf.keras.models.load_model('model_new.h5')

def output(person):
    x, y, w, h = person['box']
    face_img = frame[y:y+h, x:x+w]
    # im = Image.fromarray(face_img, 'RGB')
    # im = im.resize((50, 50))
    # img_array = np.array(im)
    # img_array = tf.cast(img_array, tf.float32)
    # # normalized=resized/255.0
    # # reshaped=np.reshape(normalized,(1,150,150,3))
    # # img_array = np.array(mormalized)
    # reshaped = np.expand_dims(img_array, axis=0)
    # # reshaped = np.vstack([reshaped])
    new_img=cv2.resize(face_img,(50,50))
    new_img=new_img.reshape(-1,50,50,1)
    result=model.predict(new_img)
    print(np.argmax(result))
    print('****')
    # result = model.predict(reshaped)[0][0]
    # if(result < 0.5):
    #     result = 0
    # else:
    #     result = 1
    # label=result
    cv2.rectangle(frame,
                  (x, y), (x+w, y+h),
                  color_dict[0], 2)

labels_dict={0:'without_mask',1:'with_mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

detector = MTCNN()

cap = cv2.VideoCapture(0)
while True:
    #Capture frame-by-frame
    __, frame = cap.read()

#     #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            output(person)


#     #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()
