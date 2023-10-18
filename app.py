import streamlit as st
from app_pages.multipage import MultiPage

# Load pages
from app_pages.page_introduction import page_introduction_body


app = MultiPage(app_name="Heart-disease-analysis-and-prediction")

# Add app pages using .add_page()
app.add_page("Quick Project Introduction", page_introduction_body)

app.run()
