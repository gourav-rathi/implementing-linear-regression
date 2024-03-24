#by TEAM-DELTA
# Importing necessary libraries for data manipulation and analysis
import pandas as pd  # For handling datasets using DataFrames
# Importing necessary libraries for numerical computation
import numpy as np  # For numerical operations on arrays
# Importing necessary libraries for data visualization
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For statistical data visualization
# Importing necessary libraries for machine learning
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.linear_model import LinearRegression  # For implementing Linear Regression model
from sklearn.metrics import mean_squared_error, r2_score  # For evaluating the model performance

# Step 1: Data Cleaning
# Load the dataset into a Pandas DataFrame
df = pd.read_csv("U:\DATASETS\Dataset.csv")  # Replace "DataSet.csv" with the actual filename

# Check for missing values and handle them appropriately
missing_values = df.isnull().sum()
if missing_values.any():
    df = df.dropna()  # Removing rows with missing values, you can replace this with your preferred method
# Ensure that all data types are appropriate for analysis

# Step 2: Exploratory Data Analysis (EDA)
# Perform descriptive statistics to understand the distribution of variables
print("Descriptive statistics:\n", df.describe())

# Visualization
pairplot_fig = sns.pairplot(df)
pairplot_fig.fig.canvas.manager.set_window_title('RATHI Pairplot')
plt.show()

# Step 3: Correlation Analysis
# Calculate correlation coefficients between predictor variables and target variable
plt.figure().canvas.manager.set_window_title('RATHI Correlation')
correlation_matrix = df.corr()
print("Correlation matrix:\n", correlation_matrix)

# Visualize correlations using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 4: Implementing Linear Regression Algorithm
# Separate the dataset into predictor variables (X) and the target variable (y)
X = df[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]
y = df['Sales ($)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement the linear regression algorithm using Ordinary Least Squares (OLS) method
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Step 5: Evaluation of the Model
# Predict the target variable using the trained model on the testing set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r_squared)

# Visualize the actual vs. predicted values
plt.figure().canvas.manager.set_window_title('RATHI Actual vs Predicted')
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales ($)")
plt.ylabel("Predicted Sales ($)")
plt.title("Actual vs Predicted Sales")
plt.show()
