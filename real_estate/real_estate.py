import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.express as px


csv_file_path = "real_estate/real_estate.csv"


df = pd.read_csv(csv_file_path)





X = df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores',
        'X5 latitude', 'X6 longitude']]
y = df['Y house price of unit area'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


st.title('Predição de preços do mercado imobiliario')


st.sidebar.header('Alguns Parametros')
inputs = {}
for col in X.columns:
    inputs[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))


prediction = model.predict(pd.DataFrame(inputs, index=[0]))[0]


st.write('Preço estimado:', round(prediction, 2))


# col1, col2 = st.columns(2)
# col3, col4 = st.columns(2)
# col5, col6= st.columns(2)

fig1 = px.scatter(df, x='X2 house age', y='Y house price of unit area', trendline='ols', title='House Age vs. Price')
fig2 = px.scatter(df, x='X3 distance to the nearest MRT station', y='Y house price of unit area', trendline='ols',
                  title='Distance to MRT Station vs. Price')
fig3 = px.scatter(df, x='X4 number of convenience stores', y='Y house price of unit area', trendline='ols',
                  title='Number of Convenience Stores vs. Price')
fig4 = px.scatter(df, x='X5 latitude', y='Y house price of unit area', trendline='ols', title='Latitude vs. Price')
fig5 = px.scatter(df, x='X6 longitude', y='Y house price of unit area', trendline='ols', title='Longitude vs. Price')


coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
fig6 = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Coefficients')

st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)
st.plotly_chart(fig4)
st.plotly_chart(fig5)
st.plotly_chart(fig6)

# col1.plotly_chart(fig1)
# col2.plotly_chart(fig2)
# col3.plotly_chart(fig3)
# col4.plotly_chart(fig4)
# col5.plotly_chart(fig5)
# col6.plotly_chart(fig6)
