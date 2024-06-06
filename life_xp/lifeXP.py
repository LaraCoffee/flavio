import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.express as px


csv_file_path = "life_xp/Life_Expectancy_Data.csv"


df = pd.read_csv(csv_file_path)



df.dropna(inplace=True)

df = pd.get_dummies(df, columns=['Country', 'Status'])


X = df.drop(columns=['Life expectancy '])
y = df['Life expectancy ']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


st.title('Modelo the regressão de expectativa de vida')


st.sidebar.header('Model Inputs')
inputs = {}
for col in X.columns:
    inputs[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))


prediction = model.predict(pd.DataFrame(inputs, index=[0]))[0]


st.write('expectativa de viad prevista:', round(prediction, 2))


importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importances')
st.plotly_chart(fig)
