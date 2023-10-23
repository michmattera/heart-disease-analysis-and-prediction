import streamlit as st
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
# from src.data_management import load_heart_disease_data
from src.data_management import load_pkl_file


def page_predict_heart_disease_body():

    st.write("### Heart Disease Predictor")

    st.info(
        f"*This page contains an interface that allows the user"
        f" to interact with the pipeline and use the model to make live predictions."
        f" /n"
        f"This page completes the fulfilment of"
        f" business requirement 2."
    )

    # Load all files
    version = "v1"
    pipeline = load_pkl_file(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/xgb_pipeline.pkl")
    best_features = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_train.csv").columns.to_list()
    feat_importance_image = plt.imread(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/y_train.csv").values
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/y_test.csv").values

    # this is the original dataset
    df = pd.read_csv(
        "outputs/datasets/collection/heart.csv")
    st.write("This is the original dataset")
    st.write(df.head())

    # filtering df just with best features
    df = df.filter(best_features)
    st.write("This is the dataset using just the best features. ")
    st.write("These are the features used to train the model.")
    st.write(df.head())
