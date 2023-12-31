from src.data_management import load_heart_disease_data
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
import numpy as np
import ppscore as pps
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")


def heart_disease_body():

    # Load all files
    df = load_heart_disease_data()

    version = "v1"
    classification_report_image = plt.imread(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/classification_report_1.png")
    features_importance_plot = plt.imread(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/best_3_features_importance.png")

    # hard copied from Feature Selection notebook
    vars_to_study = ['ca', 'cp', 'exang', 'oldpeak', 'thalach', 'chol']

    # hard copied from ModelAndEvaluation notebook
    vars_to_study_two = ['ca', 'cp', 'thal']

    st.write("## Feature Selection Study")
    st.write(
        f"This page will answer the first business requirement:"
    )
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
    st.write("### Correlation Study")
    st.write(
        f"A correlation study was conducted to better understand how "
        f"the variables are correlated to the target. \n"
    )

    # Text based on "03 - " Feature selection notebook - "Conclusions" section
    st.info(
        f"Correlation analysis methods:\n"
        f"* **Spearman** \n"
        f"* **Pearson** \n"
        f"* **Pps analysis** \n"
        f" \n"
        f"After spearman and pearson 5 variable in common were selected = \n"
        f" \n"
        f"'ca', 'cp', 'exang', 'oldpeak', 'thalach' \n"
        f" \n"
        f"Another variable that had a high predictive power was considered after pps analysis: \n"
        f" \n"
        f"'chol'"
    )

    st.info(
        f"**It is indicated that patient that has suffer from heart disease usually does not have:** \n"
        f"* Typically has no ca ( Number of Major Vessels Colored by Fluoroscopy ). \n"
        f"* Typically has no exang ( exercise induce angina ). \n"
        f"* Typically has no oldpeak ( st depression induced by exercise relative to rest ). \n"
        f" \n"
        f"**It is indicated that patient that has suffer from heart disease usually does have:** \n"
        f"* High cp ( chest pain type ). With more target cases that suffer from at least a chest pain level 2. \n"
        f"* High thalach ( maximun heart rate achieved ). Starting the peak from 150 and reaaching 175 . \n"
        f"* High chol ( serum cholestoral in mg/dl ). Many patients have chol that goes from 180 to 250. \n"
    )

    # Individual plots per variable
    if st.checkbox("Variable distribution by target"):
        st.write(
            f"Here you can see the distribution of each chosen variable against the target."
            f" \n"
            f"* Target 0: No disease \n"
            f"* Target 1: Disease \n"

        )
        variable_distribution_by_target(df)

        # Individual plots per different correlation
    if st.checkbox("Correlation heatmap"):
        st.write(
            f"### Correlation and Predictive Power Score (PPS) Heatmaps"
            f" \n"
            f"Analyze the correlations and predictive power of your dataset"
            f" \n"
            f"Here you can see how the target variable are correlated "
            f"with other variables (features and target)"
            f" \n"
            f"Analyze multi colinearity, that is, how the features are correlated among themselves \n"

        )
        correlation_heatmap(df)

    st.write("---")

    # Feature Importance Study
    st.write("### Feature Importance Study")
    st.write(
        f"A feature importance study was conducted in the notebook to better understand how "
        f"the variables are correlated to the target. \n"
        f" \n"
        f"The most important variable are: **{vars_to_study_two}**"
    )

    st.info(
        f"After feature importance study 3 variables were than found =. \n"
        f" \n"
        f"'ca', 'cp', 'thal' \n"
        f" \n"
        f"Different combination of features were tried to found best performance. \n"
    )

    # Feature importance plot
    if st.checkbox("Features importance plot"):
        st.write(
            f"Here you can see the three best feature found with feature importance while training the model"
        )
        st.image(features_importance_plot)

    st.success(
        f"The first business requirement was answered with conventional data analysis. \n"
        f" \n"
        f"Final combination of features ( Using all features from the two sets ) = \n"
        f" \n"
        f"* 'cp', 'chol', 'thalach', 'exang', 'oldpeak', 'ca', 'thal' \n"
    )

    # Text based on "03 - " Feature selection notebook - "EDA" section
    df_eda = df.filter(vars_to_study + ['target'])


# function created from " Feature Selection notebook - "EDA" section
def variable_distribution_by_target(df_eda):
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
def plot_categorical(df_eda, col, target_var):
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(data=df_eda, x=col, hue=target_var,
                  order=df_eda[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{col} vs {target_var}", fontsize=20, y=1.05)
    st.pyplot(fig)


# code copied from " Feature Selection notebook - "EDA" section
def plot_numerical(df_eda, col, target_var):
    fig = plt.figure(figsize=(10, 5))
    sns.histplot(data=df_eda, x=col, hue=target_var, kde=True, element="step")
    plt.title(f"{col} vs {target_var}", fontsize=20, y=1.05)
    st.pyplot(fig)


# Correlation heatmap
def correlation_heatmap(df):

    # load data
    df = load_heart_disease_data()

    # Calculate correlations and PPS
    df_corr_pearson, df_corr_spearman, pps_matrix = calculate_corr_and_pps(df)

    # Specify your thresholds
    corr_threshold = 0.3
    pps_threshold = 0.15

    # Display heatmaps
    st.header("Spearman Correlation Heatmap")
    heatmap_corr(df_corr_spearman, threshold=corr_threshold)

    st.header("Pearson Correlation Heatmap")
    heatmap_corr(df_corr_pearson, threshold=corr_threshold)

    st.header("Predictive Power Score (PPS) Heatmap")
    st.write(
        "PPS detects linear or non-linear relationships between two columns.")
    st.write(
        "The score ranges from 0 (no predictive power) to 1 (perfect predictive power).")
    heatmap_pps(pps_matrix, threshold=pps_threshold)


# code copied from "Feature selection notebook" - "Correlation Matrix" section
def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=12):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        fig = plt.figure(figsize=figsize)  # Create the figure
        sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                    mask=mask, cmap='viridis', annot_kws={"size": font_annot},
                    linewidth=0.5)
        plt.yticks(rotation=0)  # Set y-axis tick labels rotation
        plt.ylim(len(df.columns), 0)
        st.pyplot(plt)


# code copied from "Feature selection notebook" - "Correlation Matrix" section
def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=12):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[abs(df) < threshold] = True

        fig = plt.figure(figsize=figsize)
        ax = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                         # Adjust font size
                         mask=mask, cmap='rocket_r', annot_kws={"size": font_annot},
                         linewidth=0.05, linecolor='grey')
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


# Define a function to calculate the correlations and PPS
def calculate_corr_and_pps(df):
    df_corr_spearman = df.corr(method="spearman")
    df_corr_pearson = df.corr(method="pearson")

    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='x', index='y', values='ppscore')

    return df_corr_pearson, df_corr_spearman, pps_matrix
