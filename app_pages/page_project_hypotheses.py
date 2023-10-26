import streamlit as st
import numpy as np
import pandas as pd
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_heart_disease_data
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_performance import clf_performance
from src.machine_learning.predictive_analysis import (
    predict_heart_disease)
import matplotlib.pyplot as plt


def page_hypothesis_and_validation_body():

    st.write("## Hypothesis and Validation")

    st.write(
        f"This page contains all hypothesis and how each was validated."
        f" \n"
        f" After data analysis and feature selection study we can report that: \n"
    )

    # Hypothesis 1 - CA
    st.success(
        f"Hypothesis 1: \n"
        f" \n"
        f"Patients who have suffered from heart disease typically do not have a"
        f" significant number of major vessels colored by fluoroscopy (ca)."
        f" It is hypothesized that a lower count of major vessels colored may"
        f" be associated with heart disease."
        f" \n"
        f" **Correct** : After analysis of data heart disease patients tend to have"
        f" a lower count of major vessels."
    )

    # Hypothesis 2 - Exercise-Induced Angina
    st.info(
        f"Hypothesis 2: \n"
        f" \n"
        f"Patients with heart disease typically do not experience exercise-induced angina (exang)."
        f" It is hypothesized that the absence of exercise-induced angina may be a"
        f" characteristic of heart disease patients."
    )

    # Hypothesis 3 - ST Depression
    st.info(
        f"Hypothesis 3: \n"
        f" \n"
        f"Patients with heart disease typically do not have a high ST depression induced by exercise"
        f" relative to rest (oldpeak). It is hypothesized that a lower oldpeak value may be indicative"
        f" of heart disease."
    )

    # Hypothesis 4 - Chest Pain Type
    st.success(
        f"Hypothesis 4: \n"
        f" \n"
        f"Heart disease patients tend to have a higher level of chest pain (cp). "
        f"It is hypothesized that an increase in chest pain level is associated with a higher"
        f" likelihood of heart disease. \n"
        f" \n"
        f" **Correct** : After analysis of data heart disease patients tend to have"
        f" a chest pain level of at least 2."
    )

    # Hypothesis 5 - Maximum Heart Rate
    st.success(
        f"Hypothesis 5: \n"
        f" \n"
        f"Patients with heart disease typically achieve a higher maximum heart rate (thalach),"
        f" with values starting from around 150 and reaching up to 175. It is hypothesized that a higher"
        f" maximum heart rate may be indicative of heart disease."
    )

    # Hypothesis 6 - Serum Cholesterol
    st.success(
        f"Hypothesis 6: \n"
        f" \n"
        f"Heart disease patients often have serum cholesterol levels (chol) very high."
        f" It is hypothesized that higher cholesterol levels may be associated with a higher"
        f" risk of heart disease."
        f" \n"
        f" **Correct** : After analysis of data heart disease patients tend to have"
        f" a higher cholesterol ranging from 180 to 300 mg/dl, reaching in rare occasion even 400 and 500 mg/dl."
    )
