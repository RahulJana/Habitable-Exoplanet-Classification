import streamlit as st
import pickle
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image
menu = ["Metrics without smote","Metrics with smote"]
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
    if activity == "Metrics without smote":
        st.write("                                                                         ")
        st.header("The Metric scores of each model based on General dataset (without SMOTE)")
        st.write("                                                                         ")
        model_choice = st.selectbox("Select Model", ["Logistic regression","KNN", "DecisionTree"])

        if st.button("Predict"):
            if model_choice == "KNN":
                st.write('K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories. K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.')
                loaded_model = load_model(r"C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\pkl files\knn_model.pkl")

                prediction = loaded_model.predict(single_sample)

                pred_prob = loaded_model.predict_proba(single_sample)

                image = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\knnblack.png')

            elif model_choice == "DecisionTree":
                st.write('A Decision Tree is constructed by asking a serious of questions with respect to a record of the dataset we have got. Each time an answer is received, a follow-up question is asked until a conclusion about the class label of the record. The series of questions and their possible answers can be organised in the form of a decision tree, which is a hierarchical structure consisting of nodes and directed edges. Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.')
                loaded_model = load_model(r"C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\pkl files\Decision_tree_model.pkl")

                prediction = loaded_model.predict(single_sample)

                pred_prob = loaded_model.predict_proba(single_sample)


                image = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\dtblack.png')

            else:
                st.write("Base line- logistic regression for binary classification")
                loaded_model = load_model(r"C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\pkl files\log_reg_model.pkl")

                prediction = loaded_model.predict(single_sample)

                pred_prob = loaded_model.predict_proba(single_sample)

                image = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\lrblack.png')

            st.write("Prediction (0:Non-Habitable - 1:Habitable)")
            st.write(prediction)
            st.write("Predicted Probabilities")
            st.write(pred_prob)
            st.write("Classification report")
            st.image(image)




    elif activity == "Metrics with smote":
        st.subheader("  The Metric scores of each model based on General dataset (with SMOTE -   Synthetic Minority Oversampling Technique)")
        st.write("            ")
        st.write("Imbalance classification is a type where the dataset consists of large difference between the vales in each class this imabalance causes the machine learning model to ignore these values and have poor results. Smote stands for Synthetic Minority Oversampling Technique, where the data values of these minority classes are duplicated, so that the model recognises these classes. It is a type of data augumentation process where the new data is synthesized from the existing ones.")
        st.write("                                                                                     ")
        model_choice = st.selectbox("Select Model", ["CNN","Random Forest", "XGboost","AutoMl"])

        if st.button("Predict"):
            if model_choice == "CNN":
                st.write('Architecture of the Convolutiuonal Network:'

    'Reshape Layer, Input layer;'
    '  1D convolutional layer, consisting of 10, 3x3 filters, L2 regularization and RELU activation function;'
    '  1D max pooling layer, window size - 2x2, stride - 2;'
    '  Dropout(20%);'
    '  Fully connected layer with 48 neurons and RELU activation function;'
    '  Dropout(20%);'
    '  Fully connected layer with 18 neurons and RELU activation function;'
    '  Output layer with sigmoid activation function.')
                image3 = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\CNN_cnf.png')
                image1 = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\acc_cnn.png')
                image2 = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\cnn_loss.png')

                st.image(image1,caption="Train & Test Accuracy")
                st.image(image2,caption="Train & Test Loss")
                st.image(image3)


            elif model_choice == "XGboost":
                st.write('XGboost stands for Extreme Boosting. It is a boosting algorithm which is based on Ensemble Learning Technique. It provides a wrapper class that allows models to be treated as classifiers or regressors.'
'The overall parameters that are to be judged:'
'1. General Paramaters'
'2. Booster Parameters'
'3. Learning Task Parameters'
'Basically, there are two types of boosters, but in this case we are using Tree booster because it always outperforms the linear booster')
                image1 = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\XG_boost_conf.png')
                image2 = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\xg_boost.png')
                image3 = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\xgboost.png')
                st.image(image1)
                st.image(image2)
                st.image(image3)

            elif model_choice == "Random Forest":
                st.write('A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.')
                loaded_model = load_model(r"C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\pkl files\RF_model.pkl")

                prediction = loaded_model.predict(single_sample)

                pred_prob = loaded_model.predict_proba(single_sample)

                image = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\rf_conf.png')
                st.write("Prediction (0:Non-Habitable - 1:Habitable)")
                st.write(prediction)
                st.write("Predicted Probabilities")
                st.write(pred_prob)
                st.write("Classification report")
                st.image(image)

            else:
                st.write('EvalML is an AutoML library that builds, optimizes, and evaluates machine learning pipelines using domain-specific objective functions.'

'Combined with Featuretools and Compose, EvalML can be used to create end-to-end supervised machine learning solutions.')
                image1 = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\Automl_full.png')
                image2 = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\rankings.png')
                st.image(image1)
                st.image(image2)




