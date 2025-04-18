import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# SETTINGS
IMG_SIZE = 100
data_dir = "faces/UTKFace"  # Replace with your folder name if needed

data = []
labels_age = []
labels_gender = []

valid_extensions = (".jpg", ".jpeg", ".png", ".chip.jpg")
files = os.listdir(data_dir)
print(f"📁 Total files found: {len(files)}")

for img in files:
    if not img.lower().endswith(valid_extensions):
        continue

    try:
        # Clean filename: "age_gender_race_date.jpg.chip.jpg" or "age_gender_*.jpg"
        filename = img.split('.')[0]  # '100_0_0_201701...'
        parts = filename.split('_')

        # "25_0_0_201701.jpg" into ["25", "0", "0", "201701"]
        # hence parts = ["25", "0", "0", "201701"]
        
        if len(parts) < 2:
            continue  # Skip badly named files

        age = int(parts[0])
        gender = int(parts[1])  # 0 = Male, 1 = Female

        # Read and resize image
        img_path = os.path.join(data_dir, img)
        img_array = cv2.imread(img_path)
        #array shape (height, width, 3)

        if img_array is None:
            print(f"❌ Couldn't load image: {img_path}")
            continue

        face = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        data.append(face)
        labels_age.append(age)
        labels_gender.append(gender)

    except Exception as e:
        print(f"⚠️ Skipped {img} due to error: {e}")

print(f"\n✅ Loaded images: {len(data)}")

# Convert to NumPy arrays and normalize
X = np.array(data) / 255.0

# One-hot encode gender (Male = 0, Female = 1)
y_gender = to_categorical(np.array(labels_gender), num_classes=2)

# Convert age to age group (0: child, 1: teen, 2: adult, 3: senior)
def age_group(age):
    if age <= 12:
        return 0
    elif age <= 19:
        return 1
    elif age <= 80:
        return 2
    else:
        return 3

y_age = to_categorical(np.array([age_group(a) for a in labels_age]), num_classes=4)

# Split into training and testing
X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.2)
_, _, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.2)

# ✅ Quick check
print(f"\n✅ Final dataset shapes:")
print("X_train:", X_train.shape)
print("y_gender_train:", y_gender_train.shape)
print("y_age_train:", y_age_train.shape)


#training model 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Input shape from our data
input_shape = (100, 100, 3)
input_layer = Input(shape=input_shape)

# 📐 CNN architecture
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)

# 🧑‍🦰 Gender output (2 classes)
gender_output = Dense(2, activation='softmax', name='gender')(x)

# 👶 Age group output (4 classes)
age_output = Dense(4, activation='softmax', name='age')(x)

# 🧠 Model build
model = Model(inputs=input_layer, outputs=[gender_output, age_output])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'gender': 'categorical_crossentropy', 'age': 'categorical_crossentropy'},
              metrics={'gender': 'accuracy', 'age': 'accuracy'})

model.summary()

# Make sure these vars are from your preprocessing script:
# X_train, y_gender_train, y_age_train
# X_test, y_gender_test, y_age_test

history = model.fit(
    X_train,
    {'gender': y_gender_train, 'age': y_age_train},
    validation_data=(X_test, {'gender': y_gender_test, 'age': y_age_test}),
    epochs=15,
    batch_size=32
)

plt.figure(figsize=(12, 5))

# Gender accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['gender_accuracy'], label='Train Gender Acc')
plt.plot(history.history['val_gender_accuracy'], label='Val Gender Acc')
plt.title('Gender Accuracy')
plt.legend()

# Age accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['age_accuracy'], label='Train Age Acc')
plt.plot(history.history['val_age_accuracy'], label='Val Age Acc')
plt.title('Age Group Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

model.save("hahaha.keras")

print("✅ Model saved as hahaha.h5")
