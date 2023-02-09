import streamlit as st
import pickle
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image
menu = ["Plot"]
exo_planet_test = pd.read_csv(r"C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\exoTest.csv")
X_test = exo_planet_test.drop(['LABEL'],axis=1)
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
single_sample = X_test
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


def app():
    st.title('Project Overview')
    activity = st.selectbox("Activity", menu)
    if activity == "Plot":
        st.header("Visual Analysis of the project")

        st.subheader("Distribution of planet classification")
        image = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\distribution.png')
        st.image(image, caption='Distribution of planet classification')
        st.write('This plot shows the distribution of the number of habitable v/s non habitable planets from the given dataset. From the plot it can be inferred that out of about 5000'
                 'there about 37 planets that can be classified as habitable. These values are produced by the renowned kepler telescope of NASA.')

        st.subheader("Distribution of flux values of habitable planets")
        image = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\habitable flux.png')
        st.image(image, caption='Distribution of flux values of habitable planets')
        st.write('This plot shows the distribution of a few flux values of the planets that are classified as habitable planets. From the plot it can be inferred that the values are more towards negetive . '
                 'They are classified as habitable based on "Transit method" where the values decrease when the planet moves ahead a star.')

        st.subheader("Distribution of flux values of non-habitable planets")
        image = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\non habit flux.png')
        st.image(image, caption='Distribution of flux values of non-habitable planets')
        st.write('This plot shows the distribution of a few flux values of the planets that are classified as habitable planets. From the plot it can be inferred that the values are more towards the threshold and rarely move to postive or negetive . '
            'They are classified as habitable based on "Transit method" where the values decrease when the planet moves ahead a star.')

        st.subheader("Heatmap of the flux values")
        image = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\heatmap.png')
        st.image(image, caption='Heatmap of the flux values')
        st.write('This heatmap gives a great representation of the flux values and how they are interrelated to each other in such a way that the flux values that influence the planets habitability the most so that those can be selected to improve the accuracy of the model.')

        st.subheader("Comparison Scatter plots of different models ")
        image = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\graph_acc_updated.png')
        st.image(image, caption='Scatter plot showing tha accuracy')
        st.write(
            'This scatter plot consists of all the accuracy values as data points that are plotted to find the distribution trends of various classification models used in the project.')


