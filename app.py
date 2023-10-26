import streamlit as st
from app_pages.multipage import MultiPage

# Load pages
from app_pages.page_introduction import page_introduction_body
from app_pages.page_heart_disease_study import heart_disease_body
from app_pages.page_predict_heart_disease import page_predict_heart_disease_body
from app_pages.page_model_information import model_information_body
from app_pages.page_project_hypotheses import page_hypothesis_and_validation_body


app = MultiPage(app_name="Heart-disease-analysis-and-prediction")

# Add app pages using .add_page()
app.add_page("Quick Project Introduction", page_introduction_body)
app.add_page("Feature Selection Study", heart_disease_body)
app.add_page("Heart Disease Prediction", page_predict_heart_disease_body)
app.add_page("Model Information", model_information_body)
app.add_page("Hypothesis and validation", page_hypothesis_and_validation_body)

app.run()
