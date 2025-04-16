
# 🧠 Convolutional Neural Network (CNN) — Explained

CNN is an area of **deep learning** that helps in pattern recognition.

---
 
## 🌐 Artificial Neural Network (ANN)
A standard network consisting of multiple interconnected layers.  
Each layer:
- Receives input
- Transforms it
- Passes output to the next layer

>  **CNN is a part of ANN.**
![ANN vs CNN](https://github.com/user-attachments/assets/13acebeb-db43-411b-b0eb-af9e2cb48dae)
---

##  Filters — The Core of CNN
- A filter (e.g., 3×3) is applied over the input image.
- It slides (or **convolves**) across the image checking how closely it matches local patterns.

 **Output** → A numeric array of numbers representing how closely each patch matches the filter.

Multiple filters can be applied to capture various features.

Real life eg: (Say) windows of 2 homes are different
 - With help of filters, CNN is powerful enough to claim that they both represent windows 

---

## 🔍 Pooling
Pooling combines outputs from multiple filters to:
- Reduce dimensionality
- Retain important features
- Help understand image structure

This forms the **first layer** of CNN.

> As we go deeper, filters capture **more abstract features**.

--- 
## Deeper Layers of CNN
- **2nd Layer**: Object Detection
- **3rd Layer**: High-level abstraction (e.g., faces, objects)
 > Application of filters increases with depth (as you grow into the layers.)
---
 

## 📊 Dataset Overview

```text
✅ Loaded images: 23708
```
```text 
✅ Final dataset shapes:
X_train: (18966, 100, 100,3)
```
Shape:
	•	18966 images used for training
	•	Each image is 100 x 100 pixels
	•	3 represents RGB channels


```text
y_gender_train: (18966, 2)
```
Labels for gender classification
	•	Each label is a one-hot encoded vector with 2 elements:
	•	[1, 0] → Male
	•	[0, 1] → Female

```text
y_age_train: (18966, 4)
```
Labels for age group classification
	•	Each label is also one-hot encoded:
	•	[1, 0, 0, 0] → Child
	•	[0, 1, 0, 0] → Teen
	•	[0, 0, 1, 0] → Adult
	•	[0, 0, 0, 1] → Senior

---
## ⚙️ Frameworks: Keras & TensorFlow

- **Keras**: High-level API for building models easily;
             It provides modular, easy-to-use components for building and training neural networks  especially on top of TensorFlow.

- **TensorFlow**: Backend engine by Google. It is a open-source deep learning framework that lets you build, train, and deploy machine learning models efficiently across platforms.


> Together: `tensorflow.keras` lets you build and train models efficiently.

**Tensor** = Multidimensional array (matrices, images)  
**Flow** = The way data moves through a graph of computations

 So it’s literally a flow of tensors through operations.
 
🛠️A framework is a ready-made structure of tools, libraries, and rules that helps you build software or ML models faster and more efficiently.

Examples of ML Frameworks:  
`TensorFlow`, `PyTorch`, `Scikit-learn`

---
 
## 🛠️ Building the Model

### Imports

```python
from tensorflow.keras.models import Model
```
This is the Functional API from Keras.
  It allows us to build complex models with multiple inputs/outputs
  
```python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
```
imports all the building blocks used in the CNN architecture:

- **Input**: Defines the shape of the input going into the model (e.g., (100, 100, 3) for RGB image).

- **Conv2D**: Applies 2D convolutional filters over the image. Extracts spatial features like edges, shapes, textures.
 Each layer uses filters to “see” different aspects of the image.

- **MaxPooling2D**: Reduces size, highlights key features
 (retaining only the most important features by downsampling feature maps.)

- **Flatten**: Converts 2D matrices → 1D vectors before passing into Dense layers
It’s the bridge between convolutional layers and the final prediction layers.
  It’s like unrolling all the pixels/features into a long line — so they can be passed into a Dense (fully connected) layer. 
  Feature map shape = (12, 12, 128) --> After Flatten(), it becomes: Shape = (12 * 12 * 128) = (18432,)
  
- **Dense**: Fully connected layers
The output layers of the keras model
they only accept 1D input.

Note to remember: 
    Layers are used to construct a deep learning model 
    Tensors define the data flow through the model 
---

## Regularization (Prevent Under & Over fitting)

> Too complex a network = Overfitting
 [ training a model too much can lead to overfitting sometimes ]
 [running too many epochs will lead to overfitting]
> Model does well on training but poorly on new data.
> They are the techniques used to handle the over n under fitting issues in the model 
 
### Dropout Regularization
> This tackles overfitting issues in particular
- adding dropout layer increases the performance of the deep neural netwrok 

when network is too complex as follows, the model might overfit 
![Dropout Example](https://github.com/user-attachments/assets/9d87a7fc-c577-403f-9dad-351dc5bce8e2)

```python
Dropout(0.5)  # Dropout(rate)  -- Drops 50% neurons randomly each epoch
```

Hence  randomly drop a few random neurons to simplify the network in order for the model to fit right as follows
![Dropout](https://github.com/user-attachments/assets/9b9a0f23-f623-4c60-a4f8-35465f00b2b5)
  
---

##  Optimizer: Adam

```python
from tensorflow.keras.optimizers import Adam
```

- Adaptive learning rate
- Combines benefits of **SGD + RMSprop**
-  Adjusts learning rate automatically during training.

---

