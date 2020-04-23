import cv2
import pandas as pd
from keras.models import load_model
import re
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import glob

classifier = load_model(r'Desktop\hacker_earth\lecun_normal_rmsprop.h5')

face_cascade = cv2.CascadeClassifier(r'C:\Users\anmol singh\Downloads\dissertation-master\dissertation-master\haar_cascades\tom.xml')

y_pred = []
count = 0
filename = r'C:\Users\anmol singh\Desktop\Validation'

for img in glob.glob(filename+'/*.jpg*'):
    img = load_img(img,target_size = (200,200))
    img = img_to_array(img)
    img = img.reshape(1,200,200,3)
    img = img.astype('float32')
    x = classifier.predict(img)
    
    if x[0][0] > 0.5:
        print("Angry")
        y_pred.append("angry")
    elif x[0][1] > 0.5:
        print("Happy")
        y_pred.append("happy")
    elif x[0][2] > 0.5:
        print("Sad")
        y_pred.append("sad")
    elif x[0][3] > 0.5:
        print("Surprised")
        y_pred.append("surprised")
    else:
        print("Unknown")
        y_pred.append("Unknown")
    print(x)
    count+=1
print(count)
print(y_pred)