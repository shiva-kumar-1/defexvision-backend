import os
import json
import smtplib
from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, db
from supabase import create_client, Client
from ultralytics import YOLO
from datetime import datetime
import cv2
import cloudinary
import cloudinary.uploader
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)
CORS(app)

# ---------------------- Configuration ----------------------

# Firebase Admin setup using inline JSON from environment
firebase_json = os.getenv("FIREBASE_CREDENTIAL_JSON")
if not firebase_json:
    raise ValueError("FIREBASE_CREDENTIAL_JSON environment variable not set")

firebase_dict = json.loads(firebase_json)
cred = credentials.Certificate(firebase_dict)
firebase_app = initialize_app(cred, {
    'databaseURL': os.getenv("FIREBASE_DB_URL"),
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
})

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Email
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Load YOLO model
model = YOLO("yolov8n.pt")  # replace with custom model if needed

# ---------------------- Routes ----------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to DefexVision Backend API"})

@app.route("/detect", methods=["POST"])
def detect():
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image = request.files['image']
        file_path = f"temp_{datetime.now().timestamp()}.jpg"
        image.save(file_path)

        # Load image using OpenCV
        img = cv2.imread(file_path)

        # Run YOLO detection
        results = model(img)
        labels = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
        class_names = results[0].names if hasattr(results[0], 'names') else []
        detections = [class_names[int(cls_id)] for cls_id in labels]

        # Upload image to Cloudinary
        upload_result = cloudinary.uploader.upload(file_path)
        image_url = upload_result['secure_url']

        # Prepare metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detection_data = {
            "timestamp": timestamp,
            "defects": detections,
            "image_url": image_url
        }

        # Store in Firebase Realtime DB
        db.reference('detections').push(detection_data)

        # Store in Supabase
        supabase.table("detections").insert({
            "timestamp": timestamp,
            "defects": detections,
            "image_url": image_url
        }).execute()

        # Email notification
        send_email(detections, image_url)

        # Cleanup
        os.remove(file_path)

        return jsonify({
            "status": "success",
            "defects": detections,
            "image_url": image_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------- Email Helper ----------------------

def send_email(detected_defects, image_url):
    subject = "DefexVision Alert - Defects Detected"
    body = f"Detected Defects:\n{json.dumps(detected_defects)}\n\nImage: {image_url}"

    message = MIMEMultipart()
    message["From"] = EMAIL_SENDER
    message["To"] = EMAIL_RECEIVER
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(message)
        server.quit()
    except Exception as e:
        print(f"Failed to send email: {e}")

# ---------------------- Main ----------------------

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
