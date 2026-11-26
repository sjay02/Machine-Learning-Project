import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import pickle

data=pd.read_csv("house_priceS.csv")

X=data[['Area','Bedrooms','Bathrooms','Age','Location' ]]
y=data[ 'Price']
X = pd.get_dummies(X,columns=['Location'])


X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42

)

model=RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train,y_train)

prediction=model.predict(X_test)

print("mean absolute error:",mean_absolute_error(y_test,prediction))
print("r2 score:",r2_score(y_test,prediction))


# Save model and its feature columns together
with open("model.pkl", "wb") as file:
    pickle.dump({
        "model": model,
        "columns": X.columns.tolist()
    }, file)

print("Model and columns saved successfully!")
