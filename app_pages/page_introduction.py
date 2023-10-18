import streamlit as st


def page_introduction_body():

    st.write("### Quick project introduction")

    st.write(
        f" This ia a machine learning app to allow users to accuratly predict heart disease based on a combination of features.\n"
        f" In addiction it shows to the user how these features correlate with the final prediction.\n"
    )

    st.info(
        f"### Dataset Introduction\n"
        f"The dataset used in this project is publically available on the Kaggle website,\n "
        f"it is from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V.\n"
        f"\n"
        f"It contains 76 attributes, 1025 entries, including the predicted attribute,"
        f"but all published experiments refer to using a subset of 14 of them.\n"
        f"\n"
        f"The **target** field refers to the presence of heart disease in the patient."
        f" It is integer valued 0 = no disease and 1 = disease.\n"
    )

    st.success(
        f"### Business requirements\n"
        f"1. The client is interested in understanding the patterns from the heart disease database so that\n"
        f"the client can learn the most relevant variables correlated to a positive heart disease prediction."
        f"\n"
        f"2. The client is interested in determining if a patient would suffer from heart disease or not."
    )

    # Link to README file
    st.write(
        f"* For additional information, please read the "
        f"[Project README file](https://github.com/michmattera/heart-disease-analysis-and-prediction/tree/main)."
    )
