#The code was found on stackoverflow, i adopted and converted to a function
#https://stackoverflow.com/questions/9770073/sound-generation-synthesis-with-python

import math
import pyaudio

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  

#The path of validation images
path = os.path.join('D:/General_Assembly/projetcs/Capstone Project/Validation Image')

def get_val_img(path = 'D:/General_Assembly/projetcs/Capstone Project/Validation Image/Validation Images/'):
    val_img = []
    for img in os.listdir(path):
        image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_resize = cv2.resize(image_rgb, (224, 224))
        val_img.append(image_resize)
    return np.array(val_img).reshape(-1, 224, 224, 3)

def plot_prediction(model):
    validation_images = get_val_img(path) / 255.0

    val_prediction = model.predict(validation_images)
    val_prediction_l = ['Closed Eyes' if x >= 0.5 else 'Open Eyes' for x in val_prediction]
    for i, img in enumerate(validation_images):
        print(f'Model Score: {val_prediction[i][0]}')
        plt.imshow(img, cmap = 'gray')
        plt.title(f'Predicted {val_prediction_l[i]}')
        if val_prediction_l[i] == 'Closed Eyes':
            create_alert_sound(0.5)
            plt.show()
        else:
            plt.show()
          
      
def get_eyes(path):
    img = []
    image = cv2.imread(path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for x, y, w, h  in faces:
        roi_gray = gray_img[y: y+h, x: x+w]
        roi_color = image[y: y+h, x: x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        if len(eyes) < 2:
            print('Eyes not detected')
        else:
            for ex, ey, ew, eh in eyes:
                eyes_color = roi_color[ey: ey+eh, ex: ex+ew]
                img.append(cv2.resize(eyes_color, (224, 224)))
    img_array = np.array(img)
    return img_array           
      
def get_prediction(image):
    model4 = tf.keras.models.load_model('Model4.h5')
    path = 'D:/General_Assembly/projetcs/Capstone Project/Validation Image/' + image
    img = get_eyes(path)
    img = img/255.0
    try:
        prediction = sum(model4.predict(img))/img.shape[0]
        if prediction >= 0.5:
            print(f'Model Predict Sleepy Driver (score: {prediction})')
            original_img = cv2.imread(path)
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        else:
            print(f'Model Predict Active Driver (score: {prediction})')
            original_img = cv2.imread(path)
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    except:
        pass


def create_alert_sound(sound_duration):
    PyAudio = pyaudio.PyAudio     #initialize pyaudio

    #See https://en.wikipedia.org/wiki/Bit_rate#Audio
    BITRATE = 16000     #number of frames per second/frameset.      

    FREQUENCY = 400     #Hz, waves per second, 261.63=C4-note.

    if FREQUENCY > BITRATE:
        BITRATE = FREQUENCY+100

    NUMBEROFFRAMES = int(BITRATE * sound_duration)
    RESTFRAMES = NUMBEROFFRAMES % BITRATE
    WAVEDATA = ''    

    #generating wawes
    for x in range(NUMBEROFFRAMES):
     WAVEDATA = WAVEDATA+chr(int(math.sin(x/((BITRATE/FREQUENCY)/math.pi))*127+128))    

    for x in range(RESTFRAMES): 
     WAVEDATA = WAVEDATA+chr(128)

    p = PyAudio()
    stream = p.open(format = p.get_format_from_width(1), 
                    channels = 1, 
                    rate = BITRATE, 
                    output = True)

    stream.write(WAVEDATA)
    stream.stop_stream()
    stream.close()