import os
import time
import threading
import subprocess

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib


# -----------------------------
# PROJECT FILES
# -----------------------------
MODEL_FILE = "hand_sign_model.pkl"
DATA_FILE = "hand_sign_data.csv"
SIGNS_FOLDER = "signs"

# Only show predictions above this confidence.
CONFIDENCE_THRESHOLD = 0.60

# Speak again after this many seconds if the sign changes.
SPEECH_COOLDOWN = 1.5


# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"{MODEL_FILE} not found. Run: python train_model.py"
    )

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Collect training data first."
    )

model = joblib.load(MODEL_FILE)

# Use the exact feature order used during training.
data_columns = pd.read_csv(DATA_FILE, nrows=0).columns.tolist()
feature_columns = [
    column for column in data_columns if column != "label"
]

print("Model loaded successfully.")
print("Recognized labels:", ", ".join(model.classes_))


# -----------------------------
# MEDIAPIPE HAND DETECTION
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)


# -----------------------------
# SPEECH FUNCTION
# -----------------------------
def speak_sign(text):
    """Speak a recognized sign using macOS built-in speech."""
    try:
        subprocess.run(
            ["say", text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as error:
        print("Speech error:", error)


# -----------------------------
# CAMERA SETUP
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    hands.close()
    raise RuntimeError(
        "Cannot open the built-in webcam. Check macOS camera permissions."
    )

print("\nHand Sign Recognition started!")
print("Show one of your trained hand signs.")
print("Press Q in the webcam window to quit.")


# -----------------------------
# RECOGNITION VARIABLES
# -----------------------------
last_spoken_label = None
last_speech_time = 0.0
current_label = None
display_label = "Show your hand"


# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Mirror the camera like a selfie view.
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_label = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks.
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract the same 21 x/y/z values used in training.
            values = []

            for point in hand_landmarks.landmark:
                values.extend([point.x, point.y, point.z])

            # Create features with the expected column names.
            features = pd.DataFrame(
                [values],
                columns=feature_columns
            )

            # Predict the sign.
            probabilities = model.predict_proba(features)[0]
            best_index = int(np.argmax(probabilities))

            confidence = float(probabilities[best_index])
            candidate = str(model.classes_[best_index])

            if confidence >= CONFIDENCE_THRESHOLD:
                current_label = candidate
                display_label = candidate.replace("_", " ").title()
            else:
                display_label = "Uncertain - try again"

        else:
            display_label = "Show your hand"

        # -----------------------------
        # DISPLAY PREDICTION
        # -----------------------------
        cv2.putText(
            frame,
            f"Sign: {display_label}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if current_label else (0, 0, 255),
            2
        )

        if current_label:
            cv2.putText(
                frame,
                f"Confidence: {confidence:.0%}",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            "Press Q to quit",
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Hand Sign Recognition", frame)

        # -----------------------------
        # SHOW MATCHING SIGN IMAGE
        # -----------------------------
        if current_label:
            image_path = os.path.join(
                SIGNS_FOLDER,
                current_label + ".png"
            )

            if os.path.isfile(image_path):
                sign_image = cv2.imread(image_path)

                if sign_image is not None:
                    cv2.imshow("Matching Sign Picture", sign_image)
            else:
                # No illustration added yet: show a simple text preview.
                preview = np.full(
                    (180, 420, 3), 255, dtype=np.uint8
                )

                cv2.putText(
                    preview,
                    display_label,
                    (15, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2
                )

                cv2.imshow("Matching Sign Picture", preview)

        # -----------------------------
        # SPEAK THE SIGN NAME
        # -----------------------------
        now = time.time()

        if current_label:
            should_speak = (
                current_label != last_spoken_label
                and now - last_speech_time >= SPEECH_COOLDOWN
            )

            if should_speak:
                threading.Thread(
                    target=speak_sign,
                    args=(display_label,),
                    daemon=True
                ).start()

                last_spoken_label = current_label
                last_speech_time = now
        else:
            last_spoken_label = None

        # Press Q to quit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Hand Sign Recognition stopped.")
