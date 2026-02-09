import cv2
import mediapipe as mp
import csv
import os

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

DATA_FILE = "asl_landmarks.csv"

# Create file with header if not exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

print("Dataset collection started...")
print("Press A / B / C to save samples")
print("Press Q to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    cv2.putText(img, "Press A/B/C to Save | Q to Quit", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("ASL Landmark Collector", img)

    key = cv2.waitKey(1) & 0xFF

    if key in [ord('a'), ord('b'), ord('c')] and results.multi_hand_landmarks:
        label = chr(key).upper()

        with open(DATA_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([label] + landmarks)

        print(f"Saved sample for {label}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
