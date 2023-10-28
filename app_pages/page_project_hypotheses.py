import streamlit as st
from feature_engine.discretisation import ArbitraryDiscretiser
from src.data_management import load_heart_disease_data
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_performance import clf_performance
from src.machine_learning.predictive_analysis import (
    predict_heart_disease)


def page_hypothesis_and_validation_body():

    st.write("## Hypothesis and Validation")

    st.write(
        f"This page contains all hypothesis and how each was validated or not."
        f" \n"
        f" * **Validated** hypothesis will have a **green** background."
        f" \n"
        f" * **Confutated** hypothesis will have a **yellow** background. \n"
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
        f" be associated with heart disease. \n"
        f" \n"
        f" **Correct** : After analysis of data heart disease patients tend to have"
        f" a lower count of major vessels."
    )

    # Hypothesis 2 - Exercise-Induced Angina
    st.success(
        f"Hypothesis 2: \n"
        f" \n"
        f"Patients with heart disease typically do not experience exercise-induced angina (exang)."
        f" It is hypothesized that the absence of exercise-induced angina may be a"
        f" characteristic of heart disease patients. \n"
        f" \n"
        f" **Correct** : After analysis of data heart disease patients tend to not have"
        f" experienced exang."
    )

    # Hypothesis 3 - ST Depression
    st.success(
        f"Hypothesis 3: \n"
        f" \n"
        f"Patients with heart disease typically do not have a high ST depression induced by exercise"
        f" relative to rest (oldpeak). It is hypothesized that a lower oldpeak value may be indicative"
        f" of heart disease. \n"
        f" \n"
        f" **Correct** : After analysis of data heart disease patients tend to have"
        f" lower oldpeak."
    )

    # Hypothesis 4 - Chest Pain Type
    st.success(
        f"Hypothesis 4: \n"
        f" \n"
        f"Heart disease patients tend to have a higher level of chest pain (cp)."
        f" It is hypothesized that an increase in chest pain level is associated with a higher"
        f" likelihood of heart disease. \n"
        f" \n"
        f" **Correct** : After analysis of data heart disease patients tend to have"
        f" a chest pain level of at least 2."
    )

    # Hypothesis 5 - Maximum Heart Rate
    st.success(
        f"Hypothesis 5: \n"
        f" \n"
        f"Patients with heart disease typically achieve a higher maximum heart rate (thalach)."
        f" It is hypothesized that a higher maximum heart rate is associated with a"
        f" higher likelihood of heart disease. \n"
        f"\n"
        f" **Correct** : After analysis of data heart disease patients tend to have"
        f" higher heart rate, with more patients having from 150 and reaaching 175."
    )

    # Hypothesis 6 - Serum Cholesterol
    st.success(
        f"Hypothesis 6: \n"
        f" \n"
        f"Heart disease patients often have serum cholesterol levels (chol) very high."
        f" It is hypothesized that higher cholesterol levels may be associated with a higher"
        f" risk of heart disease. \n"
        f" \n"
        f" **Correct** : After analysis of data heart disease patients tend to have"
        f" a higher cholesterol ranging from 180 to 300 mg/dl, reaching in rare occasion even 400 and 500 mg/dl."
    )

    # Hypothesis 7 - Gender
    st.warning(
        f"Hypothesis 7: \n"
        f" \n"
        f"One gender exhibits a higher susceptibility to the condition than the other. "
        f"It is hypothesized that the gender of the patient is a significant risk factor. \n"
        f" \n"
        f" **Incorrect** : After analysis, the feature 'sex' have limited predictive power in the"
        f" specific context of the problem. It means that it doesn't provide much information"
        f" that helps the model distinguish between different outcomes"
        f" (e.g., the presence or absence of heart disease)."
    )

    # Hypothesis 8 - Age
    st.warning(
        f"Hypothesis 8: \n"
        f" \n"
        f"A specific range of age exhibits a higher susceptibility to the condition than the other. "
        f"It is hypothesized that the age of the patient is a significant risk factor. \n"
        f" \n"
        f" **Incorrect** : After analysis, the feature 'age' have limited predictive power in the"
        f" specific context of the problem. It means that it doesn't provide much information"
        f" that helps the model distinguish between different outcomes"
        f" (e.g., the presence or absence of heart disease)."
        f" Even if it was discovered that patients that suffer from heart"
        f" disease often relate with a patient of age from 45 to 55."
    )
