92.5% of training accuracy for age
90% validation accuracy  for age
There is a small generalization gap in the final few epochs — indicating potential overfitting.


94.5% of training accuracy for gender
93% validation accuracy  for gender 


Validation accuracy is slightly lower, indicating slight overfitting but still generalizes well.

epochs - An epoch is one complete pass through the entire training dataset by the model.
Cross-entropy is a loss function used to measure how well a classification model’s predicted probabilities match the actual labels.

We usef categorical_crossentropy because your task is a multi-class classification problem with one-hot encoded labels. Here's a breakdown
![accuracy](https://github.com/user-attachments/assets/41ebc0a2-0197-43a8-85fc-c62c2c6fb75d)


Some examples of the prediction:
![mworks](https://github.com/user-attachments/assets/f827a248-440f-4f19-94ee-b1ac5962dee4)
![m2works](https://github.com/user-attachments/assets/b6f147f0-bcfb-4d9b-9181-4cb81ece78a5)
![fworks](https://github.com/user-attachments/assets/fc6d032c-e4f6-460d-a930-97f9896f172b)
