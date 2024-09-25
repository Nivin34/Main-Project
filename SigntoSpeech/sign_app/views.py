from django.shortcuts import render
from django.urls import include, path
from django.http import StreamingHttpResponse
from tempfile import NamedTemporaryFile
from django.http import HttpResponse
import cv2
import os, time
import tensorflow as tf
import pyttsx3
import numpy as np
from django.shortcuts import render
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
import mediapipe as mp
from django.http import HttpResponse
from .function_S import mediapipe_detection, extract_keypoints, mp_hands
# Load your trained model

def load_model():
    json_file = open("C:/Users/Nivin/Desktop/project/Main-Project/SigntoSpeech/learning_app/model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("C:/Users/Nivin/Desktop/project/Main-Project/SigntoSpeech/learning_app/model.h5")
    return model

colors = []
for i in range(0,20):
    colors.append((245,117,16))
print(len(colors))
def prob_viz(res, actions, input_frame, colors,threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
model = load_model()
actions = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','S','R','T','U','V','W','X','Y','Z'] # Define your actions or load them
last_length = 0
current_length = 0
# Generator function to capture video frame, process it and yield it
def gen(camera):
    global sentence, sentence, predictions, actions, threshold, accuracy, last_length
    with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        threshold = 0.5 
        sequence = []
        sentence = []
        accuracy=[]
        predictions = []
        last_length = 0
        while True:
            success, frame = camera.read()
            if not success:
                break
                        
            # Process frame
            # frame = process_frame(frame, model)
            cropframe=frame[40:400,0:300]
            frame=cv2.rectangle(frame,(0,40),(300,400),255,2)
            image, results = mediapipe_detection(cropframe, hands)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try: 
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    
                            else:
                                sentence.append(actions[np.argmax(res)])     

                    if len(sentence) > 20: 
                        sentence = sentence[-20:]
                     
            except Exception as e:
                pass
                
            cv2.rectangle(frame, (0,0), (640, 40), (0, 165, 16), -1)
            print(sentence)
            if (sentence) != 0 and last_length != len(sentence):
                text_to_speech_pyttsx3(''.join(sentence))
                last_length = len(sentence)
            cv2.putText(frame,"Text: -"+' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue  # If frame is not encoded, skip it
            frame = buffer.tobytes()
            yield (b'--frame\r\n'  
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  

def process_frame(frame, model):
    
    output_frame = frame.copy()  

    # Display processed frame
    cv2.putText(output_frame, "Processed Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return output_frame

def video_stream(request):
    return StreamingHttpResponse(gen(cv2.VideoCapture(0)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
def sign(request):
    
    return render(request,'translator_app/sign.html')


def text_to_speech_pyttsx3(text):
    # Initialize the converter
    engine = pyttsx3.init()
    
    # Set properties before adding anything to speak
    engine.setProperty('rate', 150)  # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1
    
    # Adding things to say
    engine.say(text) 
    engine.runAndWait()  
