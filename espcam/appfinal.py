from flask import Flask, request, Response, render_template_string
import cv2, numpy as np
import threading, time, os
import mediapipe as mp
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Optional Twilio
try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None

app = Flask(__name__)

# ---------------- CONFIG ----------------
import os

# ---------------- CONFIG ----------------
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
TO_EMAIL = os.getenv("TO_EMAIL")

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO = os.getenv("TWILIO_TO")

UPLOAD_FOLDER = "upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Mediapipe setup ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
drawer = mp.solutions.drawing_utils

# ---------------- Global latest frame ----------------
latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
latest_frame_to_process = None
frame_lock = threading.Lock()

# ---------------- HTML ----------------
HTML_PAGE = """
<html>
<body>
<h2>Hands-Up Detection</h2>
<img src="/latest.jpg" width="640" height="480">
<script>
setInterval(()=>{document.querySelector("img").src="/latest.jpg?ts="+new Date().getTime();},1000);
</script>
</body>
</html>
"""

# ---------------- ALERT FUNCTIONS ----------------
def send_email(image_path):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = "🚨 Hands-Up Alert!"
        msg['From'] = EMAIL_USER
        msg['To'] = TO_EMAIL
        msg.attach(MIMEText("Person detected with hands up! See attached image."))

        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read(), name=os.path.basename(image_path))
            msg.attach(img)

        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        print("📧 Email sent!")
    except Exception as e:
        print("❌ Email failed:", e)

def send_sms():
    if TwilioClient is None:
        print("⚠️ Twilio not installed, cannot send SMS")
        return
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_AUTH)
        message = client.messages.create(
            body="🚨 Hands-Up detected!",
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        print("📩 SMS sent! SID:", message.sid)
    except Exception as e:
        print("❌ SMS failed:", e)

# ---------------- FRAME PROCESSING ----------------
def process_frame(frame):
    global latest_frame
    ts = int(time.time())
    frame_path = os.path.join(UPLOAD_FOLDER, f"frame_{ts}.jpg")
    cv2.imwrite(frame_path, frame)  # Save every frame

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ No pose detected")
        latest_frame = frame.copy()
        return

    lm = results.pose_landmarks.landmark
    lw, rw = lm[mp_pose.PoseLandmark.LEFT_WRIST.value], lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    ls, rs = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Debug: print wrist & shoulder positions
    print(f"LW.y={lw.y:.2f}, LS.y={ls.y:.2f}, RW.y={rw.y:.2f}, RS.y={rs.y:.2f}")

    # Dynamic margin: 20% of shoulder-wrist distance
    margin_left = 0.2 * abs(ls.y - lw.y)
    margin_right = 0.2 * abs(rs.y - rw.y)
    hands_up = (lw.y + margin_left < ls.y) and (rw.y + margin_right < rs.y)
    print(f"[{time.strftime('%H:%M:%S')}] Hands-Up = {hands_up}")

    # Draw skeleton
    drawer.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    latest_frame = frame.copy()

    # Send alerts if hands-up detected
    if hands_up:
        print(f"[{time.strftime('%H:%M:%S')}] 🚨 Hands-Up detected! Sending alert...")
        threading.Thread(target=send_email, args=(frame_path,), daemon=True).start()
        threading.Thread(target=send_sms, daemon=True).start()

# ---------------- WORKER THREAD ----------------
def frame_worker():
    global latest_frame_to_process
    while True:
        if latest_frame_to_process is not None:
            with frame_lock:
                frame = latest_frame_to_process.copy()
                latest_frame_to_process = None
            process_frame(frame)
        else:
            time.sleep(0.01)  # avoid busy wait

threading.Thread(target=frame_worker, daemon=True).start()

# ---------------- FLASK ROUTES ----------------
@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/latest.jpg')
def latest():
    global latest_frame
    _, jpeg = cv2.imencode('.jpg', latest_frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file_bytes = np.asarray(bytearray(request.data), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is not None:
            with frame_lock:
                global latest_frame_to_process
                latest_frame_to_process = frame.copy()
            return "OK"
        return "Error", 500
    except Exception as e:
        print("❌ Upload failed:", e)
        return "Error", 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🚀 Server started at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
