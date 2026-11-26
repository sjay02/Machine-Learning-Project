import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import pickle

# Load data
data = pd.read_csv("customer.csv")

# Input and output
X = data[['Age','Income','Previous_Purchase']]
y = data['Buy']

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)
print("Accuracy:",accuracy)

print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        prediction
    )
)

pickle.dump(
    model,
    open("model.pkl","wb")
)

print("Model Saved")