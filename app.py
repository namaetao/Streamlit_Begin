import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Define the pages
main_page = st.Page("pages/home.py", title="Main Page", icon="🎈")
sale_predict = st.Page("pages/sale_predict.py", title="Regression", icon="❄️")
cat_dog_classify = st.Page("pages/cat_dog_classification_app.py", title="Classification", icon="🎉")
breeds_object_detection = st.Page("pages/object_detection_app.py", title="Detection", icon="🐶")
# Set up navigation
pg = st.navigation([main_page, sale_predict, cat_dog_classify,breeds_object_detection])

# Run the selected page
pg.run()