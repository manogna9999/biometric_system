from flask import Flask, render_template, Response, request, redirect, url_for, session, send_file, jsonify
import cv2
import os
import datetime
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.secret_key = 'admin_secret'

# Load trained classifier and label encoder
classifier, label_encoder = joblib.load('cnn_models/face_classifier.pkl')

# Load MobileNetV2 for feature extraction
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
face_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# OpenCV Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam
camera = cv2.VideoCapture(0)

# State variables
attendance_taken = set()
last_marked_id = None


def mark_attendance(student_id):
    today = datetime.date.today().strftime('%d-%m-%Y')
    file_path = 'Attendance/attendance.xlsx'
    os.makedirs('Attendance', exist_ok=True)

    metadata = pd.read_csv('student_database/student_data.csv')  # Assumes columns: ID, Name, Phone, College

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = metadata.copy()

    if today not in df.columns:
        df[today] = ''

    df.loc[df['ID'] == int(student_id), today] = 'Present'
    df.to_excel(file_path, index=False)


def gen_frames():
    global attendance_taken, last_marked_id

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_img = rgb_frame[y:y + h, x:x + w]
                if face_img.size == 0:
                    continue

                face_resized = cv2.resize(face_img, (224, 224))
                face_array = img_to_array(face_resized)
                face_array = preprocess_input(face_array)
                face_array = np.expand_dims(face_array, axis=0)

                embedding = face_model.predict(face_array)[0]
                pred = classifier.predict([embedding])[0]
                student_id = label_encoder.inverse_transform([pred])[0]

                if student_id not in attendance_taken:
                    mark_attendance(student_id)
                    attendance_taken.add(student_id)
                    last_marked_id = student_id

                cv2.putText(frame, f"ID: {student_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/attendance')
def attendance():
    return render_template('attendance.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_last_id')
def get_last_id():
    global last_marked_id
    if last_marked_id:
        response = jsonify({'last_id': last_marked_id})
        last_marked_id = None
        return response
    return jsonify({'last_id': ''})


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            return send_file('Attendance/attendance.xlsx', as_attachment=True)
        else:
            return render_template('admin.html', error='Invalid credentials')
    return render_template('admin.html')


if __name__ == '__main__':
    app.run(debug=True)
