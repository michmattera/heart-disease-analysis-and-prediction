import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_heart_disease_data, load_pkl_file
from src.machine_learning.evaluate_performance import clf_performance


def model_information_body():

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

    st.write("## Model Information")

    # Introduction section
    st.info(
        "Welcome to the Machine Learning Model Dashboard."
        " This dashboard provides an overview of our model and its performance."
        f"\n\n"
        "Developer tried different"
        " approaches to find best model, with higher performance."
        f"\n\n"
        "Below are listed the model information , performance, different approaches"
        " tried by the developer and plots for visual .."
    )

    # Model Information section
    st.header("Model Information")
    st.markdown("- The type of model : xgbclassifier")
    st.markdown("- Features used in the model :")
    st.write(best_features)
    st.markdown("- Training data source : ")
    st.write(X_train)
    st.markdown("- Model hyperparameters: ")

    # show pipeline steps
    st.write("* **This is the final ML pipeline used to predict a heart disease**")
    st.write(pipeline)

    # Model Performance section
    st.header("Model Performance")
    st.write("Discuss how well your model performs. Include performance metrics and evaluation results. For example:")
    st.markdown("- Accuracy, precision, recall")
    st.markdown("- Confusion matrix")

    st.write("### Pipeline Performance")
    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=pipeline,
                    label_map=["No Disease", "Disease"])

    # Issue found by the developer and solved
    st.warning("- Duplicates data were removed from model that was causing issue"
               " of performance 100 % in test and train set")
