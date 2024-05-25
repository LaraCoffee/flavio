import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.express as px

# Step 1: Read the CSV file into a pandas DataFrame

df = pd.read_csv("Life_Expectancy_Data.csv")

# Step 2: Preprocess the data
# Drop any rows with missing values
df.dropna(inplace=True)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['Country', 'Status'])

# Step 3: Build a multiple regression model
# Define features and target variable
X = df.drop(columns=['Life expectancy '])  # Features
y = df['Life expectancy ']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Create a Streamlit app
# Streamlit code
st.title('Modelo the regressão de expectativa de vida')

# Sidebar with user input options
st.sidebar.header('Model Inputs')
inputs = {}
for col in X.columns:
    inputs[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

# Predict life expectancy based on user input
prediction = model.predict(pd.DataFrame(inputs, index=[0]))[0]

# Display the prediction
st.write('expectativa de viad prevista:', round(prediction, 2))

# Step 5: Visualize the results using Plotly Express
# Plot feature importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotly express bar chart
fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importances')
st.plotly_chart(fig)
