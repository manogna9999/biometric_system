import serial
import time
import os
import csv

def collect_ecg_data(student_id, duration=30):
    """
    Collect ECG signal data from Arduino serial for a given duration (seconds)
    and save to student_database/ecg_signals/{student_id}/ecg_data.csv
    """
    ecg_dir = f'../student_database/ecg_signals/{student_id}'
    os.makedirs(ecg_dir, exist_ok=True)
    filename = f'{ecg_dir}/ecg_data.csv'

    # Change 'COM3' to your Arduino port, or '/dev/ttyUSB0' on Linux
    ser = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)  # wait for serial connection to initialize

    print(f"[INFO] Starting ECG data collection for {duration} seconds...")
    start_time = time.time()

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'ecg_value'])

        while time.time() - start_time < duration:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    ecg_val = float(line)
                    timestamp = time.time()
                    writer.writerow([timestamp, ecg_val])
                    print(f"ECG: {ecg_val}")
                except ValueError:
                    pass  # ignore invalid lines

    ser.close()
    print(f"[INFO] ECG data saved to {filename}")

if __name__ == '__main__':
    sid = input("Enter Student ID: ")
    collect_ecg_data(sid)
