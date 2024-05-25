import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.express as px



df = pd.read_csv("real_estate.csv" )




X = df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores',
        'X5 latitude', 'X6 longitude']]  # Features
y = df['Y house price of unit area']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Create a Streamlit app
# Streamlit code
st.title('Predição de preços do mercado imobiliario')

# Sidebar with user input options
st.sidebar.header('Alguns Parametros')
inputs = {}
for col in X.columns:
    inputs[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

# Predict house price based on user input
prediction = model.predict(pd.DataFrame(inputs, index=[0]))[0]

# Display the prediction
st.write('Preço estimado:', round(prediction, 2))

# Step 5: Visualize the results using Plotly Express
# You can add more visualizations here if needed
