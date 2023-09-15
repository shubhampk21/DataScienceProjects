import streamlit as st
import pandas as pd
import numpy as np
import pmdarima as pm
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

# Load your dataset
@st.cache_resource
def load_data():
    try:
        data = pd.read_csv("train.csv")  # Update with your dataset path
        return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the file path.")
        st.stop()

data = load_data()

# Set up the Streamlit app
st.title("Sales Forecasting with BATS")

# Load the pre-trained BATS model from a .pkl file
@st.cache_resource
def load_BATS_model():
    try:
        with open("Forcasting.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        st.error("model file not found. Please check the file path.")
        st.stop()

model = load_BATS_model()

# Forecast future sales
st.header("Sales Forecasting")

future_periods = st.slider("Number of future periods to forecast", 1, 365, 1)

# Forecast future sales with BATS
forecast = model.forecast(steps=future_periods)
# Assuming 'data' is your DataFrame containing the sales data
last_date = pd.to_datetime(data["date"].iloc[-1])
future_dates = [last_date + relativedelta(days=x) for x in range(1, future_periods+1)]

forecast_df = pd.DataFrame({'date': future_dates, 'sales_forecast': forecast})
forecast_df.set_index('date', inplace=True)

# Plot the forecast
st.write("### Sales Forecast Plot")
fig = px.line(forecast_df, x=forecast_df.index, y='sales_forecast', title='Sales Forecast')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Sales')
st.plotly_chart(fig)

# Show the forecasted sales values as a table
st.write("### Forecasted Sales Values")
st.dataframe(forecast_df)

forecast_df= forecast_df.to_csv(index=True).encode("utf-8")

st.download_button(label="download data", data = forecast_df, file_name='forecast_df.csv', mime='text/csv')