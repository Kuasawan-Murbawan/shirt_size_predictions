import streamlit as st

st.title("Shirt Size Prediction")

user_age = st.slider('How old are you?', 0, 100)

user_weight = st.number_input('Insert your weight')

user_height = st.number_input('Insert your height')

st.write('Your age is:', user_age, 'Height:', user_height, 'Weight:', user_weight)

import joblib
import pandas as pd

model = joblib.load('model.pkl')


sample = pd.DataFrame([[user_weight, user_age, user_height]], columns=['weight', 'age', 'height'])

output= model.predict(sample)

if st.button('Predict'):
    st.write('Your predicted size is: ', output[0])