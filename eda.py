import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import numpy as np

def run():
    #Membuat Title
    st.title('Analyzing of bank success in conducting marketing campaigns')

    #Mmebuat sub header 
    st.subheader('EDA for Analyzing Bank Marketing Campaign datasets')
    #membuat deskripsi
    st.write('page ini dibuat oleh *St. Syakirah*')
    #Menambahkan gambar
    image = Image.open('F:/Hacktiv8/Fase 1/Milestone 2/bank.jpg')
    st.image(image, caption='Marketing Campaign')

    #Membuat garis lurus
    st.markdown('---')

    #Magic syntax
    '''
    Pada page kali ini, penulis akan melakukan eksplorasi sederhana
    '''
    #show dataframe
    data = pd.read_csv('F:/Hacktiv8/Fase 1/Milestone 2/bank.csv')
    st.dataframe(data)

    #Membuat barplot
    st.write('#### Barplot Based on User Input')
    pilihan = st.radio('Select column: ', ('deposit', 'default', 'housing', 'loan', 'job', 'education'))
    fig=plt.figure(figsize=(7,7))
    sns.countplot(x=pilihan, data=data)
    st.pyplot(fig)

    # Membuat Histogram berdasarkan input user /st.radio piliha/selectbox drop down menu
    st.write('### Histogram bBsed on User Input')
    pilihan = st.radio('Select column: ', ('age', 'balance', 'distribution', 'pdays'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(data[pilihan], bins=30, kde=True)
    st.pyplot(fig)

if __name__== '__main__':
    run()