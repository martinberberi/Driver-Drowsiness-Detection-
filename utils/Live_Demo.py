import cv2
import numpy as np
import tensorflow as tf
from flask import Flask,render_template,Response
from functions import create_alert_sound as make_sound


app=Flask(__name__)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

model4 = tf.keras.models.load_model('Model4.h5')

def generate_images():

    video = cv2.VideoCapture(0)
    #if not video.isOpened():
        #video = cv2.VideoCapture(0)
    #if not video.isOpened():
        #print('An error occured with camera')
    counter = 0
    while True:
        ret, image = video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #If i want to put the face in rectangle will be used this code
        #face = face_cascade.detectMultiScale(gray, 1.1, 4)
        #for xf, yf, wf, hf in face:
            #cv2.rectangle(image, (xf, yf), (xf + wf, yf + hf), (255, 255, 0), 2)
        
        eyes = eyes_cascade.detectMultiScale(gray, 1.1, 4)
        for x, y, w, h in eyes:
            roi_gray = gray[y: y+h, x: x+w]
            roi_color = image[y: y+h, x: x+w]
            cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)
            eyess = eyes_cascade.detectMultiScale(roi_gray)
            if len(eyess) < 1:
                print('Eyes Not Detected')
    
            else:
                for (ex, ey, ew, eh) in eyess:
                    eyes_roi = roi_color[ey: ey+eh, ex: ex+ew]
                
        final_image = cv2.resize(eyes_roi, (224, 224))
        final_image = np.array(final_image).reshape(-1, 224, 224, 3)
        final_image = final_image / 255.0
        
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        
        prediction = model4.predict(final_image)
        if prediction < 0.5:
            status = 'Active Driver'
            cv2.putText(image, status, (100, 70), font, 3, (255, 255, 0), 2, cv2.LINE_8)
            
            
        else:
            counter = counter + 1
            status = 'Drowsy Driver'
            cv2.putText(image, status, (100, 70), font, 3, (255, 255, 0), 2, cv2.LINE_8)
            
            if counter > 15:
                cv2.putText(image, 'Sleep Alert!!', (125, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2, cv2.LINE_8)
                make_sound(1)
                counter = 0        
    
        cv2.imshow('Drowsiness Detection', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video.release()
    cv2.destroyAllWindows()
        
 
        
    
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video', methods=['GET', 'POST'])
def video():
    return Response(generate_images(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)
        