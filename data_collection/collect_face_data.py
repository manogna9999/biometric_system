import cv2
import os
import csv

def collect_face_data(student_id, name, phone, college):
    face_dir = f'../student_database/faces/{student_id}'
    os.makedirs(face_dir, exist_ok=True)

    # Save to student_data.csv
    csv_path = '../student_database/student_data.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Name', 'Phone', 'College'])

    # Append student details
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([student_id, name, phone, college])

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    print("[INFO] Starting face capture. Look at the camera...")

    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            img_path = f"{face_dir}/img_{count}.jpg"
            cv2.imwrite(img_path, face_img)
            count += 1
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"Capturing {count}/20", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow('Face Collector', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif count >= 20:
            break

    print(f"[INFO] Saved {count} face images for ID {student_id}")
    cam.release()
    cv2.destroyAllWindows()

# Example usage:
if __name__ == '__main__':
    sid = input("Enter Student ID: ")
    name = input("Enter Student Name: ")
    phone = input("Enter Phone Number: ")
    college = input("Enter College Name: ")
    collect_face_data(sid, name, phone, college)
