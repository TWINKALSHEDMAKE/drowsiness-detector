from flask import Flask, Response, jsonify, request
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance
import pygame
import threading
import time
import os
import winsound  # Backup for sound

app = Flask(__name__)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

# Global variables
frame_count = 0
drowsy_events = 0
alert_sent = False
current_ear = 0.0
is_detecting = False
cap = None

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

# Fix file path - use relative path
landmark_file = "shape_predictor_68_face_landmarks.dat"
if os.path.exists(landmark_file):
    predictor = dlib.shape_predictor(landmark_file)
    print("[SUCCESS] Landmark file loaded")
else:
    print(f"[ERROR] File not found: {landmark_file}")
    print("Please download and place the file in current directory")
    exit()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

pygame.mixer.init()

def play_alert_sound():
    """Play alert sound with fallback options"""
    try:
        # Try playing WAV file
        if os.path.exists("alert.wav"):
            pygame.mixer.music.load("alert.wav")
            pygame.mixer.music.play()
        else:
            # Fallback to system beep
            winsound.Beep(1000, 500)
    except Exception as e:
        print(f"Sound error: {e}")
        winsound.Beep(1000, 500)

def detection_loop():
    global frame_count, drowsy_events, alert_sent, current_ear, is_detecting, cap
    
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return
    
    print("[INFO] Detection loop started")
    
    while is_detecting:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        ear = 0.0
        face_detected = False
        
        for rect in rects:
            face_detected = True
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            current_ear = ear

            # Drowsiness detection logic
            if ear < EAR_THRESHOLD:
                frame_count += 1
                if frame_count >= CONSEC_FRAMES:
                    drowsy_events += 1
                    print(f"[ALERT] Drowsiness #{drowsy_events} - EAR: {ear:.3f}")

                    # Play alert sound
                    play_alert_sound()

                    frame_count = 0

                    if drowsy_events >= 4 and not alert_sent:
                        print("[EMERGENCY] Sending alert SMS to parents")
                        alert_sent = True
            else:
                frame_count = 0
        
        if not face_detected:
            current_ear = 0.0
            frame_count = 0

        time.sleep(0.03)

def generate_frames():
    global is_detecting, cap, current_ear, frame_count, drowsy_events
    
    if cap is None:
        cap = cv2.VideoCapture(0)
    
    while is_detecting:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add text overlay
        cv2.putText(frame, f"EAR: {current_ear:.3f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Drowsy Events: {drowsy_events}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame Count: {frame_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status = "DETECTING" if is_detecting else "STOPPED"
        color = (0, 255, 0) if is_detecting else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask Routes
@app.route('/')
def index():
    # Use the HTML content from your index.html file
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global is_detecting, frame_count, drowsy_events, alert_sent, cap
    
    if not is_detecting:
        is_detecting = True
        frame_count = 0
        drowsy_events = 0
        alert_sent = False
        
        # Start detection thread
        thread = threading.Thread(target=detection_loop)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'started', 'message': 'Detection started'})
    else:
        return jsonify({'status': 'already_running', 'message': 'Detection already running'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global is_detecting, cap
    
    is_detecting = False
    
    def release_camera():
        time.sleep(2)
        if cap:
            cap.release()
            cap = None
    
    threading.Thread(target=release_camera).start()
    
    return jsonify({'status': 'stopped', 'message': 'Detection stopped'})

@app.route('/test_alert', methods=['POST'])
def test_alert():
    try:
        play_alert_sound()
        return jsonify({'status': 'success', 'message': 'Alert sound tested'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reset_counters', methods=['POST'])
def reset_counters():
    global frame_count, drowsy_events, alert_sent
    frame_count = 0
    drowsy_events = 0
    alert_sent = False
    return jsonify({'status': 'success', 'message': 'Counters reset'})

@app.route('/get_stats')
def get_stats():
    return jsonify({
        'ear': current_ear,
        'frame_count': frame_count,
        'drowsy_events': drowsy_events,
        'is_detecting': is_detecting
    })

if __name__ == '__main__':
    print("🚀 Starting Drowsiness Detection Web Server...")
    print("📧 Open http://localhost:5000 in your browser")
    print("⚠️  Make sure shape_predictor_68_face_landmarks.dat is in same folder")
    app.run(debug=True, host='0.0.0.0', port=5000)