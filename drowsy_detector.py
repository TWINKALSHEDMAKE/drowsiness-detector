import dlib
import cv2
from imutils import face_utils
from scipy.spatial import distance
import pygame

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

frame_count = 0
drowsy_events = 0
alert_sent = False
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/Lenovo/OneDrive/Desktop/drowsy_project/shape_predictor_68_face_landmarks.dat")





(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

pygame.mixer.init()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < EAR_THRESHOLD:
            frame_count += 1
            if frame_count >= CONSEC_FRAMES:
                drowsy_events += 1
                print(f"[ALERT] Dulki #{drowsy_events}")

                # Alert sound bajao
                pygame.mixer.music.load("alert.wav")
                pygame.mixer.music.play()

                frame_count = 0

                if drowsy_events == 4 and not alert_sent:
                    print("[EMERGENCY] Sending alert SMS to parents (Add your SMS code here)")
                    alert_sent = True
        else:
            frame_count = 0

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
