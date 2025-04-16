## 🏗️ Full CNN Architecture

### 🔹 Step 1: Input

```python
input_shape = (100, 100, 3)
input_layer = Input(shape=input_shape)
```
>Defines the shape of the input image:
  100 x 100 pixels
  3 color channels (RGB)
---

### 🔹 Step 2: First Convolution Layer

```python
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
```
- Applies 32 filters of size 3x3 on the input image; 
- Each filter detects different features (edges, textures, patterns) 
- activation='relu' adds non-linearity and avoids vanishing gradients
---

### 🔹 Step 3:  First MaxPooling Layer

```python
x = MaxPooling2D((2, 2))(x)
```
 - Reduces spatial size by taking the maximum value in a 2x2 block
 - Helps reduce memory use and computation
 - Also acts as a noise reducer

---
### 🔹 Step 4: Second Convolution Layer

```python
x = Conv2D(64, (3, 3), activation='relu')(x)
```
- Applies 64 filters to the previous layer
- Captures more complex features (corners, shapes, etc.)

---
### 🔹 Step 5: Second MaxPooling Layer

```python
x = MaxPooling2D((2, 2))(x)
```
---
### 🔹 Step 6: Third Convolution Layer


```python
x = Conv2D(128, (3, 3), activation='relu')(x)
```
- Applies 128 filters
- Now the model is learning very deep features (combinations of edges, shapes, etc.)
  
---

### 🔹 Step 7: Third MaxPooling


```python
x = MaxPooling2D((2, 2))(x)
```

---


### 🔹 Step 8: Flatten()

```python
x = Flatten()(x)
```

  - Converts 3D output (10 x 10 x 128) → 1D vector of length 12800
  - This is necessary before feeding into fully connected (Dense) layers
    
---

### 🔹 Step 9: Dropout

```python
x = Dropout(0.5)(x)
```

  - During training, randomly disables 50% of the neurons
  - Prevents overfitting by making the model rely on more features

---
### 🔹 Step 10: Gender Output
```python
gender_output = Dense(2, activation='softmax', name='gender')(x)
```

  - A final layer with 2 neurons (Male & Female)
  - softmax ensures the output is a probability (e.g., [0.9, 0.1])

---
### 🔹 Step 11: Age Output

```python
age_output = Dense(4, activation='softmax', name='age')(x)
```
  - A final layer with 4 neurons (age groups: child, teen, adult, senior)
  - Also uses softmax for multi-class classification

![Final Model Diagram](https://github.com/user-attachments/assets/67031dac-93ce-4686-86f1-873f15f0eb0d)

---

> 💡 This CNN can now classify **gender (2 classes)** and **age (4 classes)** from images.

---

