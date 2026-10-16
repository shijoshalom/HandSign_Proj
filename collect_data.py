

import cv2
import mediapipe as mp
import csv
import os
import time

DATA_FILE = "hand_sign_data.csv"
TARGET_SAMPLES = 50

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot access webcam.")
    hands.close()
    raise SystemExit

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["label"] + [
                f"point_{i}_{axis}"
                for i in range(21)
                for axis in ("x", "y", "z")
            ]
        )

label = input("Enter sign label (e.g. open_palm): ").strip().lower()

if not label:
    print("Label cannot be empty.")
    camera.release()
    hands.close()
    raise SystemExit

count = 0
collecting = False
start_time = 0
saved_rows = []

print("\nAutomatic hand sign data collection")
print("C = collect 50 samples automatically")
print("N = change sign")
print("Q = quit")

while True:
    success, frame = camera.read()

    if not success:
        print("Cannot read camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks = None

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS
        )

        landmarks = []
        for point in hand.landmark:
            landmarks.extend([point.x, point.y, point.z])

    if collecting:
        elapsed = time.time() - start_time

        if landmarks is not None and count < TARGET_SAMPLES:
            saved_rows.append([label] + landmarks)
            count += 1

        if count >= TARGET_SAMPLES:
            with open(DATA_FILE, "a", newline="") as file:
                csv.writer(file).writerows(saved_rows)

            print(f"Saved {count} samples for {label}")
            collecting = False
            saved_rows = []

    status = (
        f"COLLECTING: {count}/{TARGET_SAMPLES}"
        if collecting
        else "Press C to collect samples"
    )

    cv2.putText(
        frame, f"Sign: {label}", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )

    cv2.putText(
        frame, status, (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    cv2.imshow("Automatic Hand Sign Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c") and not collecting:
        count = 0
        saved_rows = []
        start_time = time.time()
        collecting = True
        print(f"Collecting samples for {label}...")

    elif key == ord("n") and not collecting:
        new_label = input("Enter next sign label: ").strip().lower()

        if new_label:
            label = new_label
            count = 0
            print(f"Next sign: {label}")

    elif key == ord("q"):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()
print(f"Dataset saved to {DATA_FILE}")
