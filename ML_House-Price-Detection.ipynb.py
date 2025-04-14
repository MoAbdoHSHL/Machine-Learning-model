"""
# 🏡 California House Price Prediction using Linear Regression

This project demonstrates how to build a simple **machine learning model** to predict house prices using the **California Housing Dataset** provided by `scikit-learn`.

Steps:
- Load and explore real-world housing data
- Use **Linear Regression** to model the relationship between features (income, age, rooms,..) and house prices
- Evaluate the model's performance using metrics like **Mean Squared Error (MSE)** and **R² score**
"""


"""
## 📊 1. Load and Explore the Data
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
data = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedianHouseValue'] = data.target

# Show first few rows
print(df.head())

"""
## 📈 2. Visualize Feature vs Target

Here, we visualize the relationship between **Median Income** and **House Price** to see how strongly they are correlated.
"""

plt.scatter(df['MedInc'], df['MedianHouseValue'], alpha=0.3)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Income vs House Price')
plt.grid(True)
plt.show()


"""
## 🧠 3. Split and Train the Linear Regression Model
"""

X = df.drop('MedianHouseValue', axis=1)  # Features
y = df['MedianHouseValue']               # Target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)


"""
## 📊 4. Evaluate the Model

We use **Mean Squared Error (MSE)** and **R² Score** to evaluate the model’s performance.

### MSE Formula:
\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2
\]

- \( y_{\text{true}, i} \): Actual value
- \( y_{\text{pred}, i} \): Predicted value
- \( n \): Number of observations

### R² Formula:
\[
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2}{\sum_{i=1}^{n} (y_{\text{true}, i} - \bar{y})^2}
\]

- \( \bar{y} \): Mean of actual values
"""

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


"""
## 🎯 5. Visualize Actual vs Predicted Results

A perfect model would predict all points exactly on the red dashed line.
"""

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3, color='teal')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.plot([0, 5], [0, 5], 'r--')  # 45-degree line
plt.grid(True)
plt.show()

