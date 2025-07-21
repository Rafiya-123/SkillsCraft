import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Generate dummy image data
def generate_image(label):
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    cv2.putText(img, label, (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

X, y = [], []
for i in range(10):
    X.append(generate_image("Cat").flatten())
    y.append(0)
    X.append(generate_image("Dog").flatten())
    y.append(1)

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict and report
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
