import streamlit as st
from app_pages.multipage import MultiPage

# Load pages
from app_pages.page_introduction import page_introduction_body
from app_pages.page_heart_disease_study import heart_disease_body


app = MultiPage(app_name="Heart-disease-analysis-and-prediction")

# Add app pages using .add_page()
app.add_page("Quick Project Introduction", page_introduction_body)
app.add_page("Heart Disease Study", heart_disease_body)

app.run()
