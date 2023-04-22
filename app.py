import numpy as np
from flask import Flask, render_template, Response,send_from_directory,  request, session, redirect, url_for, send_file, flash,g
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import os
import imutils

CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
#CLASSES = ['motorbike', 'person', 'helmet']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model=tf.keras.models.load_model('new_helmet_model.h5')
helmet_model = cv2.dnn.readNetFromCaffe('helmet_model.prototxt', 'helmet_model.caffemodel')
motorbike_model = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

app=Flask(__name__)
uploads_dir = "uploads/"
app.config['UPLOAD_FOLDER'] = uploads_dir
app.config['SECRET_KEY'] = 'the random string'
filename=""

def detect_helmet(frame, startX, startY, endX, endY):
    helmet = False
    try:
        roi = frame[startY:endY, startX:endX]
        blob = cv2.dnn.blobFromImage(cv2.resize(roi, (224, 224)), 1, (224, 224), (104, 117, 123))
        helmet_model.setInput(blob)
        detections = helmet_model.forward()
         
        if detections[0, 0, 0, 2] > 0.5:
            helmet = True
    except:
        pass
    return helmet

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            color = [int(c) for c in COLORS[classIds[i]]]
            # green --> bike
            # red --> number plate
            if classIds[i]==0: #bike
                helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
            else: #number plate
                x_h = x-60
                y_h = y-350
                w_h = w+100
                h_h = h+100
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                # h_r = img[max(0,(y-330)):max(0,(y-330 + h+100)) , max(0,(x-80)):max(0,(x-80 + w+130))]
                if y_h>0 and x_h>0:
                    h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                    c = helmet_or_nohelmet(h_r)
                    cv2.putText(img,['helmet','no-helmet'][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)                
                    cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 10)

def get_video(path,name):
    cap = cv2.VideoCapture(path)
    ret=True
    while ret:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600, height=600)
        try:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            motorbike_model.setInput(blob)
            detections = motorbike_model.forward()
            persons = []
            person_roi = []
            motorbi = []
            
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # filter out weak detections by ensuring the confidence
                # is greater than minimum confidence
                if confidence > 0.5:
                    
                    # extract index of class label from the detections
                    idx = int(detections[0, 0, i, 1])
                    
                    if idx == 15:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        # roi = box[startX:endX, startY:endY/4] 
                        # person_roi.append(roi)
                        persons.append((startX, startY, endX, endY))
    
                    if idx == 14:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        motorbi.append((startX, startY, endX, endY))
    
            xsdiff = 0
            xediff = 0
            ysdiff = 0
            yediff = 0
            p = ()
            
            for i in motorbi:
                mi = float("Inf")
                for j in range(len(persons)):
                    xsdiff = abs(i[0] - persons[j][0])
                    xediff = abs(i[2] - persons[j][2])
                    ysdiff = abs(i[1] - persons[j][1])
                    yediff = abs(i[3] - persons[j][3])
    
                    if (xsdiff+xediff+ysdiff+yediff) < mi:
                        mi = xsdiff+xediff+ysdiff+yediff
                        p = persons[j]
                        # r = person_roi[j]
    
    
                if len(p) != 0:
    
    	            # display the prediction
    	            label = "{}".format(CLASSES[14])
    	            print("[INFO] {}".format(label))
    	            cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), COLORS[14], 2)
    	            y = i[1] - 15 if i[1] - 15 > 15 else i[1] + 15
                    
    	            cv2.putText(frame, label, (i[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[14], 2)   
    	            label = "{}".format(CLASSES[15])
    	            print("[INFO] {}".format(label))
    
    	            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), COLORS[15], 2)
    	            y = p[1] - 15 if p[1] - 15 > 15 else p[1] + 15
    
    	            roi = frame[p[1]:p[1]+(p[3]-p[1])//4, p[0]:p[2]]
    	            print(roi)
    	            if len(roi) != 0:
    	            	img_array = cv2.resize(roi, (50,50))
    	            	gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    	            	img = np.array(gray_img).reshape(1, 50, 50, 1)
    	            	img = img/255.0
    	            	prediction = model.predict_proba([img])
    	            	cv2.rectangle(frame, (p[0], p[1]), (p[0]+(p[2]-p[0]), p[1]+(p[3]-p[1])//4), COLORS[0], 2)
    	            	cv2.putText(frame, str(round(prediction[0][0],2)), (p[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)
    
        except:
            pass
    
        _,buffer=cv2.imencode('.jpg',frame)
        x=buffer.tobytes()

        yield(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + x + b'\r\n')
        
def get_cam():
    cap = cv2.VideoCapture(0)
    ret=True
    while ret:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600, height=600)
        try:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward() 
            persons = []
            person_roi = []
            motorbi = []
            
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # filter out weak detections by ensuring the confidence
                # is greater than minimum confidence
                if confidence > 0.5:
                    
                    # extract index of class label from the detections
                    idx = int(detections[0, 0, i, 1])
                    
                    if idx == 15:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        # roi = box[startX:endX, startY:endY/4] 
                        # person_roi.append(roi)
                        persons.append((startX, startY, endX, endY))
    
                    if idx == 14:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        motorbi.append((startX, startY, endX, endY))
    
            xsdiff = 0
            xediff = 0
            ysdiff = 0
            yediff = 0
            p = ()
            
            for i in motorbi:
                mi = float("Inf")
                for j in range(len(persons)):
                    xsdiff = abs(i[0] - persons[j][0])
                    xediff = abs(i[2] - persons[j][2])
                    ysdiff = abs(i[1] - persons[j][1])
                    yediff = abs(i[3] - persons[j][3])
    
                    if (xsdiff+xediff+ysdiff+yediff) < mi:
                        mi = xsdiff+xediff+ysdiff+yediff
                        p = persons[j]
                        # r = person_roi[j]
    
    
                if len(p) != 0:
    
    	            # display the prediction
    	            label = "{}".format(CLASSES[14])
    	            print("[INFO] {}".format(label))
    	            cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), COLORS[14], 2)
    	            y = i[1] - 15 if i[1] - 15 > 15 else i[1] + 15
    	            cv2.putText(frame, label, (i[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[14], 2)   
    	            label = "{}".format(CLASSES[15])
    	            print("[INFO] {}".format(label))
    
    	            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), COLORS[15], 2)
    	            y = p[1] - 15 if p[1] - 15 > 15 else p[1] + 15
    
    	            roi = frame[p[1]:p[1]+(p[3]-p[1])//4, p[0]:p[2]]
    	            print(roi)
    	            if len(roi) != 0:
    	            	img_array = cv2.resize(roi, (50,50))
    	            	gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    	            	img = np.array(gray_img).reshape(1, 50, 50, 1)
    	            	img = img/255.0
    	            	prediction = model.predict_proba([img])
    	            	cv2.rectangle(frame, (p[0], p[1]), (p[0]+(p[2]-p[0]), p[1]+(p[3]-p[1])//4), COLORS[0], 2)
    	            	cv2.putText(frame, str(round(prediction[0][0],2)), (p[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)
    
        except:
            pass
    
        _,buffer=cv2.imencode('.jpg',frame)
        x=buffer.tobytes()

        yield(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + x + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        value=request.form['select']
        if value=="cam":
            return render_template("webcam.html");
        else:
            if 'upFile' not in request.files:
                flash('No file part')
                return render_template("download.html",text="no file part")
            file = request.files['upFile']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return render_template("download.html",text="no selected file")
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return render_template("download.html",text=filename);
    return render_template("index.html");

@app.route('/video')
def video():
    file=os.listdir(app.config['UPLOAD_FOLDER'])
    filename=file[0]
    return Response(get_video(os.path.join(app.config['UPLOAD_FOLDER'], filename),filename),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam')
def cam():
    return Response(get_cam(),mimetype='multipart/x-mixed-replace; boundary=frame')
           

if __name__=="__main__":
    app.run()