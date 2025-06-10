import os
import cv2
import uuid
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import firebase_admin
from firebase_admin import credentials, storage, db
from supabase import create_client, Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# === Load environment variables ===
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CREDENTIAL_PATH")
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUD_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUD_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# === Setup Flask ===
app = Flask(__name__)
CORS(app)

# === Setup YOLO model ===
model = YOLO("models/best.pt")
defect_classes = ["IC-defect", "LED-defect", "Mouse-click defect", "Mouse-scrolldefect", "Resistor-defect", "capacitor-defect"]

# === Setup Cloudinary ===
cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=CLOUD_API_KEY,
    api_secret=CLOUD_API_SECRET
)

# === Setup Firebase ===
cred = credentials.Certificate(FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred, {
    'storageBucket': FIREBASE_STORAGE_BUCKET,
    'databaseURL': FIREBASE_DB_URL
})

# === Setup Supabase ===
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Email function ===
def send_email(subject, body, receiver=EMAIL_RECEIVER):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = receiver
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("[📧] Email sent")
    except Exception as e:
        print(f"[❌] Email failed: {e}")

# === Helper: Upload to Cloudinary ===
def upload_image_to_cloudinary(image_path):
    try:
        result = cloudinary.uploader.upload(image_path)
        return result.get("secure_url")
    except Exception as e:
        print(f"[❌] Cloudinary Upload Failed: {e}")
        return None

# === Route: Upload & detect image ===
@app.route("/upload", methods=["POST"])
def upload():
    try:
        image = request.files["image"]
        email = request.form.get("email", EMAIL_RECEIVER)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}.jpg"
        image_path = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        image.save(image_path)

        results = model(image_path)

        for r in results:
            img = r.orig_img.copy()
            detected_classes = []

            for box in r.boxes:
                cls_id = int(box.cls)
                class_name = r.names[cls_id]
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                category = "Defect" if class_name in defect_classes else "Non-Defect"
                color = (0, 0, 255) if category == "Defect" else (0, 255, 0)
                label = f"{class_name} ({conf:.2f})"
                cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
                cv2.putText(img, label, (coords[0], coords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                detected_classes.append(class_name)

            result_path = os.path.join("results", f"result_{timestamp}.jpg")
            os.makedirs("results", exist_ok=True)
            cv2.imwrite(result_path, img)

            url = upload_image_to_cloudinary(result_path)

            # Save to Firebase
            ref = db.reference("detections")
            ref.push({
                "timestamp": timestamp,
                "classes": detected_classes,
                "url": url,
                "email": email
            })

            # Save to Supabase
            supabase.table("detections").insert({
                "timestamp": timestamp,
                "classes": detected_classes,
                "image_url": url,
                "email": email
            }).execute()

            # Email user
            send_email(
                subject="⚠️ Defect Detection Result",
                body=f"Timestamp: {timestamp}\nDetected: {detected_classes}\nImage URL: {url}",
                receiver=email
            )

        return jsonify({"message": "Detection complete", "url": url, "classes": detected_classes})
    except Exception as e:
        print(f"[❌] Upload error: {e}")
        return jsonify({"error": str(e)}), 500

# === Home route ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to DefexVision Backend API"})

# === Run the app ===
if __name__ == "__main__":
    app.run(debug=True)
