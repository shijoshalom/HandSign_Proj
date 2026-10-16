import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Cannot access the webcam.")
else:
    print("Webcam opened successfully!")
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        cv2.imshow("Mac Webcam Test - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
