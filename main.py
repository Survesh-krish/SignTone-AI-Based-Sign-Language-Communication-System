# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
from camera1 import VideoCamera1
import mysql.connector
import hashlib
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import imagehash
from werkzeug.utils import secure_filename
from PIL import Image
import argparse
import urllib.request
import urllib.parse

import torch
#from transformers import AutoTokenizer, AutoModel
import speech_recognition as sr

import pygame
import time

from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="sign_tone"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    f2=open("lang.txt","w")
    f2.write("")
    f2.close()
    
    return render_template('index.html',msg=msg)




@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!' 
   
    return render_template('login.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']

        mycursor.execute("SELECT max(id)+1 FROM register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO register(id,name,mobile,email,uname,pass) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,email,uname,pass1)
        mycursor.execute(sql,val)
        mydb.commit()
        return redirect(url_for('login_user'))

    
        
    return render_template('register.html',msg=msg)

@app.route('/admin', methods=['GET', 'POST'])
def admin():    
    dimg=[]
    '''path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #resize
        img = cv2.imread('static/data/'+fname)
        rez = cv2.resize(img, (300, 300))
        cv2.imwrite("static/dataset/"+fname, rez)'''        
        
    return render_template('admin.html',dimg=dimg)

@app.route('/train_gesture', methods=['GET', 'POST'])
def train_gesture():
    msg=""
    mycursor = mydb.cursor()

    
    
    if request.method=='POST':
        gname=request.form['gname']
        
        
        
        mycursor.execute("SELECT count(*) FROM ga_gesture where gesture=%s",(gname,))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM ga_gesture")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            ff=open("static/label.txt","w")
            ff.write(gname)
            ff.close()
            gf="f"+str(maxid)

            
            gfile="f"+str(maxid)+".csv"
            ff=open("static/label1.txt","w")
            ff.write(gfile)
            ff.close()
                    
            sql = "INSERT INTO ga_gesture(id,gesture,fname) VALUES (%s, %s, %s)"
            val = (maxid,gname,gfile)
            mycursor.execute(sql,val)
            mydb.commit()
        else:
            mycursor.execute("SELECT * FROM ga_gesture where gesture=%s",(gname,))
            gd = mycursor.fetchone()
            gid=gd[0]
            ff=open("static/label.txt","w")
            ff.write(gname)
            ff.close()
            gf="f"+str(gid)

            
            gfile="f"+str(gid)+".csv"
            ff=open("static/label1.txt","w")
            ff.write(gfile)
            ff.close()    
        msg="ok"
    
        
    return render_template('train_gesture.html',msg=msg)

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    msg=""
    act=request.args.get("act")
    st=request.args.get("st")
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM ga_gesture")
    gdata = mycursor.fetchall()

    if st=="del":
        did=request.args.get("did")
        mycursor.execute("SELECT * FROM ga_gesture where id=%s",(did,))
        gd = mycursor.fetchone()
        gfile=gd[2]
        os.remove("static/hand_gesture_data/"+gfile)
        mycursor.execute("delete from ga_gesture where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('capture',act='1'))
        
        
    return render_template('capture.html',msg=msg,act=act,gdata=gdata)

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ga_gesture")
    data = mycursor.fetchall()

    dt=[]
    dt2=[]
    for dc in data:
        dt.append(dc[1])
        d1=dc[2].split(".")
        dt2.append(d1[0])
        
    cname="|".join(dt)
    cname2="|".join(dt2)
    ff=open("static/class1.txt","w")
    ff.write(cname)
    ff.close()

    ff=open("static/class2.txt","w")
    ff.write(cname2)
    ff.close()
    
    #build model
    DATA_DIR = "static/hand_gesture_data"

    # Load data
    data = []
    labels = []
    gesture_map = {}  # Label mapping

    for idx, file in enumerate(os.listdir(DATA_DIR)):
        gesture_name = file.split(".")[0]
        gesture_map[idx] = gesture_name  # Store label mapping

        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(file_path, header=None)
        data.extend(df.values)
        labels.extend([idx] * len(df))

    # Convert to numpy array
    X = np.array(data)
    y = np.array(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and gesture mapping
    joblib.dump(model, "gesture_model.pkl")
    joblib.dump(gesture_map, "gesture_map.pkl")

    print(f"Model trained with accuracy: {model.score(X_test, y_test) * 100:.2f}%")

    return render_template('classify.html',msg=msg,data=data)



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    msg=""
    dimg=[]
    mycursor = mydb.cursor()
    
    if request.method=='POST':
        
        message=request.form['message']
        file = request.files['file']

        mycursor.execute("SELECT max(id)+1 FROM sign_image")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        fn="F"+str(maxid)+".gif"
        file.save(os.path.join("static/upload", fn))
        
        sql = "INSERT INTO sign_image(id,message,image_file) VALUES (%s, %s, %s)"
        val = (maxid,message,fn)
        mycursor.execute(sql,val)
        mydb.commit()
        msg="success"
        #return redirect(url_for('login_user'))

    
    '''path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #resize
        img = cv2.imread('static/data/'+fname)
        rez = cv2.resize(img, (300, 300))
        cv2.imwrite("static/dataset/"+fname, rez)'''
        
        
    return render_template('upload.html',msg=msg)

@app.route('/view_image', methods=['GET', 'POST'])
def view_image():
    msg=""
    act=request.args.get("act")
    dimg=[]
    mycursor = mydb.cursor()
    
    mycursor.execute("SELECT * FROM sign_image")
    data = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")

        mycursor.execute("SELECT * FROM sign_image where id=%s",(did,))
        d1 = mycursor.fetchone()
        fn=d1[2]
        if os.path.exists("static/upload/"+fn):
            os.remove("static/upload/"+fn)
        mycursor.execute("delete from sign_image where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('view_image'))

        
    return render_template('view_image.html',msg=msg,data=data,act=act)

@app.route('/test_voice', methods=['GET', 'POST'])
def test_voice():
    msg=""
    st=""
    vtext=""
    act=request.args.get("act")
    dimg=[]
    mycursor = mydb.cursor()
    
    img=""

    if request.method=='POST':        
        mess=request.form['message']

        if mess=="":
            s=1
        else:
            mm="%"+mess+"%"

            mycursor.execute("SELECT * FROM sign_image where message like %s",(mm,))
            dat = mycursor.fetchall()

            for dat1 in dat:
                img=dat1[2]
                

            if img=="":
                s=1
            else:
                ff=open("static/det.txt","w")
                ff.write(mess)
                ff.close()
                ff=open("static/img.txt","w")
                ff.write(img)
                ff.close()
                return redirect(url_for('test_voice',act='1'))

    if act=="1":
        ff=open("static/det.txt","r")
        vtext=ff.read()
        ff.close()

        ff=open("static/img.txt","r")
        img=ff.read()
        ff.close()
        
    return render_template('test_voice.html',msg=msg,act=act,img=img,st=st,vtext=vtext)



#Gesture Recognition - Transformer-Based Gesture Encoder
class TransformerGestureRecognizer:
    def __init__(self, model_path='transformer_gesture_model.pt', tokenizer_name='bert-base-uncased'):
        # Load pretrained transformer model
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # MediaPipe for hand & pose detection
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

    def preprocess_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def extract_features(self, frame):
        # Initialize MediaPipe
        hands = self.mp_hands.Hands(static_image_mode=False)
        pose = self.mp_pose.Pose(static_image_mode=False)
        
        hand_results = hands.process(frame)
        pose_results = pose.process(frame)
        
        features = []
        
        # Extract hand keypoints
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
        
        # Extract pose keypoints
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        
        return np.array(features, dtype=np.float32)

    def predict_sign(self, features):
        # Convert to tensor
        input_tensor = torch.tensor(features).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_tensor)
        predicted_id = torch.argmax(logits, dim=1).item()
        # Convert ID to text (placeholder)
        return f"Sign_{predicted_id}"

#Speech Recognition and Text Conversion [RNN-Transducer and CTC]
def extract_features(audio_path, sample_rate=16000, n_mels=80):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # Convert waveform to log-Mel spectrogram
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=400,
        hop_length=160,
        win_length=400
    )(waveform)
    
    log_mel = T.AmplitudeToDB()(mel_spectrogram)
    return log_mel.transpose(1, 2)

#RNN
class RNNTEncoder():
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RNNTEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out  # Shape: (batch, time, hidden*2)

class RNNTDecoder():
    def __init__(self, vocab_size, hidden_dim):
        super(RNNTDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, y):
        emb = self.embedding(y)
        out, _ = self.lstm(emb)
        logits = self.fc(out)
        return logits

class RNNTModel():
    def __init__(self, input_dim, enc_hidden, dec_hidden, vocab_size, enc_layers=3):
        super(RNNTModel, self).__init__()
        self.encoder = RNNTEncoder(input_dim, enc_hidden, enc_layers)
        self.decoder = RNNTDecoder(vocab_size, dec_hidden)
        self.fc_joint = nn.Linear(enc_hidden*2 + dec_hidden, vocab_size)  # joint network

    def forward(self, x, y):
        enc_out = self.encoder(x)  # (batch, time, enc_hidden*2)
        dec_out = self.decoder(y)  # (batch, seq_len, dec_hidden)
        
        # Simple joint network: expand and sum
        enc_exp = enc_out.unsqueeze(2)  # (batch, time, 1, enc_hidden*2)
        dec_exp = dec_out.unsqueeze(1)  # (batch, 1, seq_len, dec_hidden)
        joint = torch.cat((enc_exp.expand(-1,-1,dec_exp.size(2),-1),
                           dec_exp.expand(-1,enc_exp.size(1),-1,-1)), dim=-1)
        logits = self.fc_joint(joint)
        return logits  # (batch, time, seq_len, vocab_size)

#CTC-Based Decoding
def ctc_decode(logits, blank=0):
   
    pred = torch.argmax(logits, dim=-1)
    prev = blank
    output = []
    for p in pred:
        if p != blank and p != prev:
            output.append(p.item())
        prev = p
    return output

class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize_speech(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Speech Recognition service unavailable"

#DNN Transcription Refinement
class DNNRefiner():
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(DNNRefiner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def refine_transcription(ctc_output_ids, id2char, refiner_model):
    # Convert IDs to one-hot embedding for DNN refinement
    vocab_size = len(id2char)
    one_hot = torch.zeros(len(ctc_output_ids), vocab_size)
    for i, idx in enumerate(ctc_output_ids):
        one_hot[i][idx] = 1.0
    
    refined_logits = refiner_model(one_hot)
    refined_ids = torch.argmax(refined_logits, dim=-1)
    refined_text = ''.join([id2char[i.item()] for i in refined_ids])
    return refined_text

     
def model_speech():
    id2char = {0: '-', 1:'a', 2:'b', 3:'c', 4:' '}  # extend as needed
    vocab_size = len(id2char)

    # Extract features
    features = extract_features("speech.wav")

    # Initialize models
    rnnt_model = RNNTModel(input_dim=80, enc_hidden=128, dec_hidden=128, vocab_size=vocab_size)
    refiner = DNNRefiner(input_dim=vocab_size, hidden_dim=64, vocab_size=vocab_size)

    # Dummy decoder input (start token)
    y = torch.zeros(1, 10).long()  # (batch, seq_len)

    # Forward pass
    logits = rnnt_model(features, y)  # (batch, time, seq_len, vocab)
    # Reduce for CTC decoding: take time dimension
    ctc_output_ids = ctc_decode(logits[0, :, 0, :])
    
    # DNN refinement
    final_text = refine_transcription(ctc_output_ids, id2char, refiner)
    print("Final Recognized Text:", final_text)

##
class TextToSpeech:
    def __init__(self, lang='en'):
        self.lang = lang

    def speak_text(self, text):
        tts = gTTS(text=text, lang=self.lang)
        filename = "output_speech.mp3"
        tts.save(filename)
        os.system(f"start {filename}")  
    
@app.route('/test_cam', methods=['GET', 'POST'])
def test_cam():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("lang.txt","r")
    lg=f2.read()
    f2.close()

    if request.method=='POST':
        lg=request.form['language']
        f2=open("lang.txt","w")
        f2.write(lg)
        f2.close()

    
        
    return render_template('test_cam.html',msg=msg,lg=lg)

def lg_translate(lg,output):
    result=""
    recognized_text=output
    recognizer = sr.Recognizer()
    translator = Translator()
    try:
        available_languages = {
            'ta': 'Tamil',
            'hi': 'Hindi',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'te': 'Telugu',
            'mr': 'Marathi',
            'ur': 'Urdu',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'fr': 'French'
        }

        print("Available languages:")
        for code, language in available_languages.items():
            print(f"{code}: {language}")

        #selected_languages = input("Enter the language codes (comma-separated) you want to translate to: ").split(',')
        selected_languages=lg.split(',')
       
        for lang_code in selected_languages:
            lang_code = lang_code.strip()
            if lang_code in available_languages:
                translated = translator.translate(recognized_text, dest=lang_code)
                print(f"Translation in {available_languages[lang_code]} ({lang_code}): {translated.text}")

                result=translated.text
               

            else:
                print(f"Language code {lang_code} not available.")

        
    except Exception as e:
        print("An error occurred during translation:", e)

    return result
    ###

####
def translate_text(text, source_language, target_language):
    api_key = 'AIzaSyDW9tvaQUsywmaILt73Go8Fy5mU6ILOixU'  # Replace with your API key
    url = f'https://translation.googleapis.com/language/translate/v2?key={api_key}'
    payload = {
        'q': text,
        'source': source_language,
        'target': target_language,
        'format': 'text'
    }
    response = requests.post(url, json=payload)
    translation_data = response.json()
    translated_text = translation_data
    #translation_data['data']['translations'][0]['translatedText']
    return translated_text

def speak(audio):
    engine = pyttsx3.init()
    engine.say(audio)
    engine.runAndWait()

def text_to_speech(text, language='en'):
    # Create a gTTS object
    tts = gTTS(text=text, lang=language, slow=False)

    # Save the audio file
    tts.save("static/output.mp3")

    # Play the audio
    #os.system("start output.mp3")  # For Windows, use "start", for macOS use "afplay", for Linux use "mpg321"

#
def Recognition():
    cap = cv2.VideoCapture(0)
    gesture_recognizer = TransformerGestureRecognizer()
    speech_module = SpeechToText()
    tts_module = TextToSpeech()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess and extract features
        frame_rgb = gesture_recognizer.preprocess_frame(frame)
        features = gesture_recognizer.extract_features(frame_rgb)

        # Predict sign language
        sign_text = gesture_recognizer.predict_sign(features)
        cv2.putText(frame, f"Sign: {sign_text}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ISL Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # speech recognition
            spoken_text = speech_module.recognize_speech()
            print("Spoken Text:", spoken_text)
            tts_module.speak_text(spoken_text)


#Avatar-Based Sign Language Synthesis
class Avatar:
    def __init__(self):
        self.head_pos = (400, 150)
        self.body_start = (400, 200)
        self.body_end = (400, 400)
        self.hand_left = (350, 250)
        self.hand_right = (450, 250)
        self.expression = "neutral"  # neutral, happy, sad, etc.

    def draw(self, screen):
        # Draw head
        pygame.draw.circle(screen, FACE_COLOR, self.head_pos, 40)
        # Draw body
        pygame.draw.line(screen, AVATAR_COLOR, self.body_start, self.body_end, 8)
        # Draw arms
        pygame.draw.line(screen, HAND_COLOR, self.body_start, self.hand_left, 6)
        pygame.draw.line(screen, HAND_COLOR, self.body_start, self.hand_right, 6)
        # Draw facial expression
        self.draw_face(screen)

    def draw_face(self, screen):
        x, y = self.head_pos
        if self.expression == "neutral":
            pygame.draw.line(screen, (0,0,0), (x-10, y+10), (x+10, y+10), 2)
        elif self.expression == "happy":
            pygame.draw.arc(screen, (0,0,0), (x-15, y, 30, 20), 3.14, 0, 2)
        elif self.expression == "sad":
            pygame.draw.arc(screen, (0,0,0), (x-15, y+10, 30, 20), 0, 3.14, 2)


def play_sign_sequence(avatar, sequence):
    """
    sequence = list of frames, each frame contains:
    {'left_hand': (x,y), 'right_hand': (x,y), 'expression': 'neutral/happy/sad'}
    """
    for frame in sequence:
        avatar.hand_left = frame['left_hand']
        avatar.hand_right = frame['right_hand']
        avatar.expression = frame['expression']
        screen.fill(BG_COLOR)
        avatar.draw(screen)
        pygame.display.update()
        clock.tick(FPS)

    # ------------------------------
    example_sequence = [
        {'left_hand': (350, 250), 'right_hand': (450, 250), 'expression': 'neutral'},
        {'left_hand': (340, 230), 'right_hand': (460, 230), 'expression': 'happy'},
        {'left_hand': (330, 210), 'right_hand': (470, 210), 'expression': 'happy'},
        {'left_hand': (340, 230), 'right_hand': (460, 230), 'expression': 'neutral'},
        {'left_hand': (350, 250), 'right_hand': (450, 250), 'expression': 'neutral'},
    ]

    # ------------------------------
    avatar = Avatar()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Play sign
        play_sign_sequence(avatar, example_sequence)

##

@app.route('/test_pro3', methods=['GET', 'POST'])
def test_pro3():
    msg=""
    fn=""
    st=""
    lfile=""
    word=""
    val=""
    
    act=request.args.get("act")
    f2=open("lang.txt","r")
    lgg=f2.read()
    f2.close()

    f3=open("static/detect.txt","r")
    ms=f3.read()
    f3.close()

    ff=open("static/class1.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split('|')

    if ms=="":
        st=""
    else:
        st="1"
        n=0
        for cc in cname:
            n+=1
            if cc==ms:              
                
                break
        print("value=")
        print(str(n))
        m=n-1
        pos=n
        ##
        
        #lfile="a"+str(pos)+"_"+lgg+".jpg"

        c=0
        if lgg=="" or lgg=="en":
            c=1
            val=ms
            word=ms
            #text_to_speech(word)
        else:
            val=lg_translate(lgg,ms)
            word=val
            #text_to_speech(word,lgg)

        ff=open("static/detect.txt","w")
        ff.write("")
        ff.close()
        

    return render_template('test_pro3.html',msg=msg,st=st,lgg=lgg,fn=fn,act=act,lfile=lfile,word=word,val=val)


############
def gen1(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed1')
def video_feed1():
    return Response(gen1(VideoCamera1()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
############
def gen(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


