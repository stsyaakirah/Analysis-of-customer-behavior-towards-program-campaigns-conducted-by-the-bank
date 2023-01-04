import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import numpy as np
import joblib
import json

#Load all files
with open('model_tree.pkl', 'rb') as file_1:
  model_tree = joblib.load(file_1)
  
with open('model_scaler.pkl', 'rb') as file_2:
  model_scaler = joblib.load(file_2)

with open('model_encoder.pkl','rb') as file_3:
  model_encoder = joblib.load(file_3)

with open('list_num_cols.txt','r') as file_4:
  list_num_cols = json.load(file_4)
  
with open('list_cat_cols.txt','r') as file_5:
  list_cat_cols = json.load(file_5)

with open('list_X_final.txt', 'r') as file_6:
  list_X_final = json.load(file_6)
  
def run():
    #Membuat Form 
    with st.form(key='form_parameters'):
        age = st.number_input('age', min_value=18, max_value=95, value=25)
        day = st.number_input('day', min_value=1, max_value=31, value=20)
        campaign = st.number_input('campaign', min_value=1, max_value=65, value=25)
        previous = st.number_input('previous', min_value=0, max_value=60, value=25)
        pdays = st.number_input('pdays', min_value=16, max_value=60, value=25)
        duration = st.number_input('duration', min_value=0, max_value=5000, value=1000)
        balance = st.number_input('balance', min_value=-7000, max_value=100000, value=500)
        st.markdown('---')
                
        job = st.selectbox('job',  ('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'), index=1)
        marital = st.selectbox('marital',  ('divorced','married','single'), index=1)        
        education = st.selectbox('education',  ('primary', 'secondary', 'tertiary', 'unknown'), index=1)
        default = st.selectbox('default',  ('no','yes'), index=1)        
        housing = st.selectbox('housing',  ('no','yes'), index=1)
        loan = st.selectbox('loan',  ('no','yes'), index=1)
        contact = st.selectbox('contact',  ('cellular','telephone', 'unknown'), index=1)   
        month = st.selectbox('month',  ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep','oct', 'nov', 'dec'), index=1)         
        poutcome = st.selectbox('poutcome',  ('failure','nonexistent','success'), index=1)
            
        st.markdown('---')
        submitted = st.form_submit_button('Predict')

    data_inf={
        'age' : age,
        'day' : day,
        'campaign' : campaign,
        'previous' : previous,
        'pdays': pdays, 
        'duration': duration,
        'balance': balance,
        'job': job,
        'marital' : marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan' : loan, 
        'contact' : contact,
        'month' : month,
        'poutcome': poutcome,
        }
    
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
    
        # Split between Numerical Columns and Categorical Columns
        X_inf_num = data_inf[list_num_cols]
        X_inf_cat = data_inf[list_cat_cols]
        print('X_inf_num', X_inf_num)
        print('X_inf_cat', X_inf_cat)
        
        # Feature Scaling and Feature Encoding
        X_inf_scaled = model_scaler.transform(X_inf_num)
        X_inf_encoded = model_encoder.transform(X_inf_cat)

        # Concate numerik and categoric
        X_inf_ = np.concatenate([X_inf_scaled, X_inf_encoded], axis=1)
            
        #Buat dataframe
        X_inf_df = pd.DataFrame(X_inf_, columns=[list_num_cols + list_cat_cols])

        # Pilih fiture yang akan dipakai dalam pemodelan
        X_inf_final = X_inf_df[list_X_final]
            
        # Predict using Decision Tree
        y_pred_tree = model_tree.predict(X_inf_final)

        st.write('#Deposit: ', str((y_pred_tree)))

if __name__== '__main__':
    run()