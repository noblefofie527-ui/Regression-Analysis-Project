import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Creating Data
data = {
    'Square_Feet': [1500, 1800, 2100, 2400, 2700, 3000, 3300, 3500, 4000, 4500],
    'Price': [300000, 340000, 390000, 450000, 480000, 550000, 590000, 620000, 700000, 780000]
}
df = pd.DataFrame(data)
df.to_csv('housing_data.csv', index=False)

# Splitting into X (Feature) and y (Target)
X = df[['Square_Feet']]
y = df['Price']

# Splitting into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialising and Fitting
model = LinearRegression()
model.fit(X_train, y_train)

# Making Predictions
y_pred = model.predict(X_test)

# Evaluating and Interpreting
print(f"Slope(m): {model.coef_[0]:.2f}")
print(f"y-intercept(b): {model.intercept_:.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.4f}")

# Visualisation
import matplotlib.pyplot as plt

# Plotting the training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')

# Plotting the regression line
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Line of Best Fit')

plt.title('House Price Prediction')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.legend()
plt.show()