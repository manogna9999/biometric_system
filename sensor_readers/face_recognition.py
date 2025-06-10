import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import datetime
import pandas as pd

# ✅ Correct model path (remove ../ if running from project root)
model = load_model('cnn_models/face_model.h5')

# ✅ Prepare label map using folder structure
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = 'student_database/faces/'

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    data_dir,
    target_size=(100, 100),
    batch_size=1,
    class_mode='categorical')

label_map = {v: k for k, v in generator.class_indices.items()}

# ✅ Load student info (CSV file with ID, Name, Phone)
student_df = pd.read_csv('student_database/student_data.csv')  # Adjust path if needed

# ✅ Attendance tracking
present_students = set()

def recognize_face():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            img = cv2.resize(face_img, (100, 100))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img, verbose=0)
            class_idx = np.argmax(pred)

            student_id = label_map.get(class_idx, "Unknown")

            if student_id != "Unknown" and student_id not in present_students:
                present_students.add(student_id)
                print(f"[INFO] Marked Present: {student_id}")

            name_text = f'ID: {student_id}'
            cv2.putText(frame, name_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Face Recognition - Press Q to Quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # ✅ Save attendance
    save_attendance()

def save_attendance():
    today = datetime.date.today().strftime("%Y-%m-%d")
    attendance_file = f'attendance/{today}_attendance.xlsx'

    attendance_list = []
    for student_id in present_students:
        row = student_df[student_df['ID'] == int(student_id)].iloc[0]
        attendance_list.append({
            "ID": row['ID'],
            "Name": row['Name'],
            "Phone": row['Phone'],
            "College": row['College'],
            "Status": "Present"
        })

    all_ids = set(student_df['ID'].astype(str))
    absentees = all_ids - present_students

    for student_id in absentees:
        row = student_df[student_df['ID'] == int(student_id)].iloc[0]
        attendance_list.append({
            "ID": row['ID'],
            "Name": row['Name'],
            "Phone": row['Phone'],
            "College": row['College'],
            "Status": "Absent"
        })

    df = pd.DataFrame(attendance_list)
    os.makedirs('attendance', exist_ok=True)
    df.to_excel(attendance_file, index=False)
    print(f"[INFO] Attendance saved to {attendance_file}")

if __name__ == '__main__':
    recognize_face()
