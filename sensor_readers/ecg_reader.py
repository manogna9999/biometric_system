import serial

def ecg_recognition():
    ser = serial.Serial('COM3', 9600, timeout=1)
    print("[INFO] Place ECG sensor")

    while True:
        line = ser.readline().decode('utf-8').strip()
        if line.startswith("ID:"):
            student_id = line.split(":")[1]
            print(f"ECG recognized: {student_id}")
            return student_id

if __name__ == '__main__':
    ecg_recognition()
