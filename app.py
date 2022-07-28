from pycaret.regression import load_model, predict_model
import streamlit as st
from readline import set_pre_input_hook
from sys import setprofile
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
import plotly.express as px




st.set_page(page_title='Survival Prediction')




def predict_survival(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]



model = load_model('deaths')


st.title('COVID gravity prediction')
st.write('This is a predictor of risk incurred when getting COVID 19')

age = st.sidebar.selectbox('Age',('0-17','18-49','50-64','65+'))

sex = st.sidebar.selectbox('Sex',('Male','Female'))

state = st.sidebar.selectbox('State',('SC', 'MN', 'NY', 'AL', 'IA', 'MD', 'TN', 'CO', 'IN', 'AR', 'KY',
       'WI', 'NM', 'MS', 'CA', 'VA', 'MA', 'NE', 'NC', 'LA', 'NH', 'OK',
       'MO', 'OH', 'NJ', 'GA', 'UT', 'KS', 'IL', 'MI', 'DE', 'FL', 'SD',
       'TX', 'WA', 'WY', 'PA', 'AK', 'ND', 'OR', 'ME', 'CT', 'AZ', 'ID',
       'NV', 'WV', 'HI', 'VT', 'MT', 'DC', 'PR', 'GU'))

county = st.sidebar.selectbox('County',(''))

race = st.sidebar.selectbox('Race',('White',
       'American Indian/Alaska Native', 'Black', 'Asian',
       'Multiple/Other', 'Native Hawaiian/Other Pacific Islander'))


features = {'age':age, 'sex':sex, 'state':state, 'race':race}


df = pd.DataFrame(features, index = [0])
prediction = predict_survival(model, df)       
features_df  = pd.DataFrame([features])




st.table(features_df)

if st.button('Predict'):    
    prediction = predict_survival(model, features_df)
    if prediction == True:
        st.write('The risk to your safety is high. Please take precautions and try avoiding infection'+ str(prediction))
    else:
        st.write('The risk is not as high. Please help ensure safety to those vulnerable around you.')