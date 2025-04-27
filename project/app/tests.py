from django.test import TestCase

# Create your tests here.
# save_model.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump

# Dataset
X = np.array([
    [1, 80, 15], [0, 30, 25], [1, 70, 18], [0, 40, 22],
    [1, 87, 4], [0, 35, 26], [1, 90, 10], [1, 85, 12],
    [0, 40, 24], [0, 45, 23], [1, 92, 8], [0, 50, 20],
    [1, 80, 13], [0, 33, 27], [1, 75, 17], [1, 95, 5],
    [0, 28, 29], [0, 50, 19], [1, 82, 11], [0, 40, 23],
    [1, 88, 7], [1, 75, 15], [0, 55, 22]
])
y = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0])

# Train and save model
model = LogisticRegression()
model.fit(X, y)
dump(model, 'weather_model.joblib')
print("Model saved successfully!")