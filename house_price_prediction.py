import pandas as pd
from sklearn.linear_model import LinearRegression

# Data load karna
data = pd.read_csv("house_data.csv")

# Features aur Target select karna
X = data[['Area']] 
y = data['Price']

# Model train karna
model = LinearRegression()
model.fit(X, y)

# Prediction (Warning se bachne ke liye DataFrame use kiya)
test_data = pd.DataFrame([[2000]], columns=['Area'])
prediction = model.predict(test_data)

print(f"Predicted House Price for 2000 sq ft: {prediction[0]:.2f}")