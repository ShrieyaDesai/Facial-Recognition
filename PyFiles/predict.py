import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === SETTINGS ===
IMG_SIZE = 100
MODEL_PATH = "hahaha.keras"  # path to your trained model
AGE_LABELS = ["Child", "Teen", "Adult", "Senior"]
GENDER_LABELS = ["Male", "Female"]

# === Load Model ===
model = load_model(MODEL_PATH)
print("✅ Model loaded!")

# === Load Image ===
image_path = input("📷 Enter path to your image (JPG/PNG): ")
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found. Check the path.")
    exit()

# Resize and normalize
resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
normalized = resized / 255.0
reshaped = np.expand_dims(normalized, axis=0)  # shape (1, 100, 100, 3)

# === Predict ===
gender_pred, age_pred = model.predict(reshaped)

gender = GENDER_LABELS[np.argmax(gender_pred)]
age = AGE_LABELS[np.argmax(age_pred)]

# === Show Result ===
print("\n🔮 Prediction Result:")
print(f"🧑 Gender: {gender}")
print(f"📅 Age Group: {age}")

# Optional: Show image with OpenCV
cv2.putText(img, f"{gender}, {age}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
