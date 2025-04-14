# Machine-Learning-model
# 🏡 California House Price Prediction using Linear Regression

This project demonstrates how to build a simple **machine learning model** to predict house prices using the **California Housing Dataset** provided by `scikit-learn`.

It is working as following:
- Load and explore real-world housing data
- Use **Linear Regression** to model the relationship between features (income, age, rooms,..) and house prices
- Evaluate the model's performance using metrics like **Mean Squared Error (MSE)** and **R² score**
  #### Formula for MSE:

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2
\]

Where:
- \( y_{\text{true}, i} \) = Actual value (the true house price)
- \( y_{\text{pred}, i} \) = Predicted value (the price predicted by the model)
- \( n \) = Number of data points (or observations)

---

### 2. **R² Score (R-Squared)**

The **R² Score** tells us how well our model explains the variance in the actual data. An R² score of **1** means the model explains all the variation in the data, while a score of **0** means the model doesn't improve upon the mean.

#### Formula for R² Score:

\[
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2}{\sum_{i=1}^{n} (y_{\text{true}, i} - \bar{y})^2}
\]

Where:
- \( y_{\text{true}, i} \) = Actual value (house price)
- \( y_{\text{pred}, i} \) = Predicted value (model's predicted price)
- \( \bar{y} \) = Mean of the actual values (average house price)

---

### 🏠 Visualizing the Metrics:

The following example shows how MSE and R² affect the model's performance:
- **MSE**: The smaller the MSE, the better the model's predictions are (closer to actual values).
- **R²**: The closer R² is to 1, the better the model explains the variation in the data.

- If the dots are close to the line, the MSE is low, and the R² is high.
- If the dots are far from the line, the MSE is high, and the R² is low.
- Visualize the relationship between actual and predicted prices
