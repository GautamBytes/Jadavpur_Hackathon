import argparse
import io
from typing import Counter
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, abort, render_template, request, redirect, send_file, session, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
from ultralytics import YOLO

# List of vehicle classes
VEHICLE_CLASSES = [
    'car', 'bike', 'auto', 'rickshaw', 'cycle', 'bus', 'minitruck', 'truck',
    'van', 'taxi', 'motorvan', 'toto', 'train', 'boat', 'cycle van'
]

# Global variables to store statistics
vehicle_counts = Counter()
total_vehicles = 0
average_speed = None

app = Flask(__name__)

@app.route("/")
def hello_world():
    dark_mode = session.get('dark_mode', False)
    return render_template('index.html', 
                           dark_mode=dark_mode,
                           vehicle_counts={},
                           total_vehicles=0,
                           average_speed=None)

from flask import send_from_directory

# Add this route to allow downloading images
@app.route("/download/image")
def download_image():
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path + '/' + latest_subfolder
    files = os.listdir(directory)
    latest_file = files[0]

    return send_from_directory(directory, latest_file, as_attachment=True)


# Add this route to allow downloading videos
@app.route("/download/video/<video_type>")
def download_video(video_type):
    folder_path = os.getcwd()
    if video_type == 'original':
        video_file = 'original_output.mp4'
    else:
        video_file = 'detected_output.mp4'
    return send_from_directory(folder_path, video_file, as_attachment=True)

# Modify the predict_img function to return the filename
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            f.save(filepath)
            
            file_extension = f.filename.rsplit('.', 1)[1].lower()
            
            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                model = YOLO('yolov9c.pt')
                detections = model(img, save=True)
                return display(f.filename)

                
            elif file_extension == 'mp4':
                process_video(filepath)
                return display(f.filename)
    
    dark_mode = session.get('dark_mode', False)
    return render_template('index.html', 
                           dark_mode=dark_mode,
                           vehicle_counts=dict(vehicle_counts), 
                           total_vehicles=total_vehicles, 
                           average_speed=average_speed)

def process_image(img):
    global vehicle_counts, total_vehicles
    
    model = YOLO('yolov9c.pt')
    results = model(img)
    
    vehicle_counts.clear()
    total_vehicles = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = box.cls
            class_name = model.names[int(c)]
            if class_name in VEHICLE_CLASSES:
                vehicle_counts[class_name] += 1
                total_vehicles += 1

def process_video(video_path):
    global vehicle_counts, total_vehicles, average_speed
    
    cap = cv2.VideoCapture(video_path)
    model = YOLO('yolov9c.pt')
    
    vehicle_counts.clear()
    total_vehicles = 0
    total_speed = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                c = box.cls
                class_name = model.names[int(c)]
                if class_name in VEHICLE_CLASSES:
                    vehicle_counts[class_name] += 1
                    total_vehicles += 1
                    
                    # Simulate speed calculation (replace with actual speed estimation if available)
                    simulated_speed = np.random.randint(20, 80)
                    total_speed += simulated_speed
        
        frame_count += 1
        
    cap.release()
    
    if frame_count > 0:
        average_speed = total_speed / frame_count
    else:
        average_speed = None

@app.route("/video_feed_original")
def video_feed_original():
    return Response(get_frame('original_output.mp4'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed_detected")
def video_feed_detected():
    return Response(get_frame('detected_output.mp4'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = os.path.join(folder_path, latest_subfolder)
    print("printing directory: ", directory) 
    files = os.listdir(directory)
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    
    print(latest_file)

    # Get the file extension of the latest file, not the input filename
    file_extension = latest_file.rsplit('.', 1)[1].lower()

    if file_extension in ['jpg', 'jpeg', 'png']:      
        return send_from_directory(directory, latest_file)
    else:
        return "Invalid file format"
        
        
@app.route("/feedback_form")
def feedback_form():
    return render_template('feedback.html')

@app.route("/feedback", methods=["POST"])
def feedback():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')

    # Save feedback to a file or database
    with open('feedback.txt', 'a') as f:
        f.write(f"{datetime.datetime.now()} - {name} ({email}): {message}\n")

    return redirect(url_for('hello_world'))        

def get_frame(video_file):
    video = cv2.VideoCapture(video_file)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")

    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO('yolov9c.pt')
    app.run(host="0.0.0.0", port=args.port)






