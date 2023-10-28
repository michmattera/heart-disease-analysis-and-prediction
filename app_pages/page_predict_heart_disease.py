import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_engine.discretisation import ArbitraryDiscretiser
from src.data_management import load_heart_disease_data
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_performance import clf_performance
from src.machine_learning.predictive_analysis import (
    predict_heart_disease)


def page_predict_heart_disease_body():

    # Load all files
    version = "v3"
    pipeline = load_pkl_file(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/xgbclassifier_pipeline.pkl")
    best_features = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_train.csv").columns.to_list()
    classification_report_image = plt.imread(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/classification_report_3.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/y_train.csv").values
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/y_test.csv").values

    st.write("## Heart Disease Predictor")

    st.info(
        f"* This page contains an interface that allows the user"
        f" to interact with the pipeline and use the model to make live predictions. \n"
    )

    st.success(
        f"Second business requirement was answered building a ML model that"
        f" predict if a patient will suffer from heart disease or not ."
    )

    # show pipeline
    st.write("---")
    st.write("#### There is 1 main Pipeline")
    st.write(" * The pipeline combine cleaning and feature engineering steps")
    st.write(" Model was trained with combination of features described"
             " in the feature selection study.")
    st.write(pipeline)

    st.write("---")

    # Insert 6 features to predict live
    st.write("### Predict if the patient will suffer or not from heart disease.  \n")
    st.info("* Please enter the 6 features needed for prediction.")

    # create input fields for live data
    X_live = InsertLiveData()
    # predict on live data
    if st.button("Run Predictive Analysis"):
        predict_heart_disease(
            X_live, best_features, pipeline)


def InsertLiveData():

    df = load_heart_disease_data()
    percentageMin, percentageMax = 0.4, 2.0

    # we create input widgets for 6 features
    # 'cp', 'chol','exang', 'oldpeak', 'ca', 'thal'
    col1, col2, col3 = st.beta_columns(3)
    col4, col5, col6 = st.beta_columns(3)

    # create an empty DataFrame, which will contain live data
    X_live = pd.DataFrame([], index=[0])

    with col1:
        feature = 'cp'
        streamlit_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            key="cp_selectbox"
        )
    X_live[feature] = streamlit_widget

    with col2:
        feature = 'chol'
        streamlit_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median(),
            key="chol_number_input"
        )
    X_live[feature] = streamlit_widget

    with col3:
        feature = 'exang'
        streamlit_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            key="exang_selectbox"
        )
    X_live[feature] = streamlit_widget

    with col4:
        feature = 'oldpeak'
        streamlit_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median(),
            key="oldpeak_number_input"
        )
    X_live[feature] = streamlit_widget

    with col5:
        feature = 'ca'
        streamlit_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            key="ca_selectbox"
        )
    X_live[feature] = streamlit_widget

    with col6:
        feature = 'thal'
        streamlit_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            key="thal_selectbox"
        )
    X_live[feature] = streamlit_widget

    return X_live
