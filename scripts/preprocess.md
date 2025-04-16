# 🧼 Preprocessing Overview in Machine Learning

In ML, preprocessing refers to:

- Reading & cleaning raw data  
- Normalizing inputs  
- Transforming labels  
- Resizing/cropping  
- Converting to a format suitable for model input  

---

## 📌 In this project

Libraries such as `cv2` and `numpy` were used to read images and convert them to numpy arrays.

**Image resizing to 100x100** was done as CNNs expect a fixed input shape. All images must be the same size, or the model won’t train.

We also initialized data containers where:
- `data`: processed images  
- `labels_age`: ages extracted from filenames  
- `labels_gender`: 0 = Male, 1 = Female  

---

### 🚫 Filters Non-image Files

Skips files that don't end with `.jpg`, `.png`, `.jpeg`, or `.chip.jpg`, hence:
- Prevents processing of `.DS_Store`, `.csv`, or any unwanted files in the folder.

---

### 🧾 Extract Metadata from Filename

```python
filename = img.split('.')[0]
parts = filename.split('_')
age = int(parts[0])
gender = int(parts[1]) 
```
Splits filename like `"25_0_0_201701.jpg"` into `["25", "0", "0", "201701"]`  
thereby extracting:  
- `age = 25`  
- `gender = 0` → male
As the UTKFace dataset encodes age and gender in the filename itself.
---

## Append to Lists

```python
data.append(face)
labels_age.append(age)
labels_gender.append(gender)
```

This builds our training dataset in memory before converting it to arrays.

---


## Convert data[] to NumPy and Normalize

Converts list of images to a NumPy array (`X`)  
Divides by 255 to normalize pixel values from `[0–255]` → `[0.0–1.0]`

---

## One-Hot Encode Gender

Machine learning does not understand how to classify the categories (e.g., jeans > tshirt)  
It works with numbers, but if jeans = 3 and tshirt = 2  
we can say 3 > 2 indirectly implying that jeans > tshirt  

This approach uses dummy variables:  
- They are an array of `n-1` size, where `n` = no. of categories  
- The array is filled with all zeroes except for the categorical place  

```python
y_gender = to_categorical(np.array(labels_gender), num_classes=2)
```

---
## Map Age to Classes - `age_group()`
---


## One-Hot Encode Age Classes

Converts age class labels `[0, 2, 1, 0]` → one-hot encoded vectors  
Format: `[[1,0,0,0], [0,0,1,0], [0,1,0,0], ...]`

```python
y_age = to_categorical(np.array([age_group(a) for a in labels_age]), num_classes=4)
```


 



