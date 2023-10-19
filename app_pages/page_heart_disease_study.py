import streamlit as st
import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_heart_disease_data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def heart_disease_body():

    # load data
    df = load_heart_disease_data()

    # hard copied from Feature Selection notebook
    vars_to_study = ['ca', 'cp', 'exang', 'oldpeak', 'thalach', 'chol']

    st.write("### Heart Disease Study")
    st.info(
        f"* The client is interested in understanding the patterns from the heart disease database "
        f"so that the client can learn the most relevant variables correlated "
        f"to a positive heart disease prediction.")

    # inspect data
    if st.checkbox("Inspect Heart Disease Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 5 rows.")

        st.write(df.head(5))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to the target. \n"
        f"Target 1: Disease \n"
        f"Target 2: No disease \n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    # Text based on "03 - " Feature selection notebook - "Conclusions" section
    st.info(
        f"After correlation analysis with pearson and spearman 5 important features where selected = \n"
        f"'ca', 'cp', 'exang', 'oldpeak', 'thalach' \n"
        f"After pps heatmap analysis another important feature was found 'chol' \n"
        f"It is indicated that patient usually does not have: \n"
        f"*Typically has no ca ( Number of Major Vessels Colored by Fluoroscopy ). \n"
        f"Typically has no exang ( exercise induce angina ). \n"
        f"Typically has no oldpeak ( st depression induced by exercise relative to rest ). \n"
        f"It is indicated that patient usually does have: \n"
        f"High cp ( chest pain type ). With more target cases that suffer from at least a chest pain level 2. \n"
        f"High thalach ( maximun heart rate achieved ). Starting the peak from 150 and reaaching 175 . \n"
        f"High chol ( serum cholestoral in mg/dl ). Many patients have chol that goes from 180 to 250. \n"


    )

    # Text based on "03 - " Feature selection notebook - "EDA" section
    df_eda = df.filter(vars_to_study + ['target'])


# function created from " Feature Selection notebook - "EDA" section
def sale_price_per_variable(df_eda):
    target_var = 'target'
    vars_to_study = ['ca', 'cp', 'exang', 'oldpeak', 'thalach', 'chol']
    for col in vars_to_study:
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
            print("\n\n")
        else:
            plot_numerical(df_eda, col, target_var)
            print("\n\n")


# code copied from " Feature Selection notebook - "EDA" section
def plot_categorical(df, col, target_var):
    fig, axes = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df, x=col, hue=target_var,
                  order=df[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)


def plot_numerical(df, col, target_var):
    fig, axes = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step")
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)
