import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('my_leaf_disease_model.h5')

# Set the background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: 'green';
    }
    </style>
    """,
    unsafe_allow_html=True
)   

# Title of the app
st.title("🍃 Leaf Disease Prediction 🍃")

# Instructions
st.write("Upload a leaf image to predict its disease.")

# Upload button
uploaded_file = st.file_uploader("Choose a leaf image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Resize and preprocess the image
    img = image.resize((224, 224))  # Resize the image as required by your model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the disease
    prediction = model.predict(img_array)
    disease_name = np.argmax(prediction, axis=1)  # Assuming your model returns class indices

    # You can map the indices to actual disease names if you have a dictionary for that
    disease_dict = {0: 'Apple_scab', 1: 'Apple_Black_rot', 2: 'Cedar_apple_rust',3:'Healthy_apple',4:'Healthy_Blueberry',5:'Cherry_(including_sour)_healthy',6:'Cherry_(including_sour)_Powdery_mildew',7:'Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot',8:'Corn_(maize)_Common_rust',9:'Corn_(maize)_healthy',10:'Corn_(maize)_Northern_Leaf_Blight',11:'Grape_Black_rot',12:'Grape_Esca_(Black_Measles)',13:'Healthy_grapes',14:'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)',15:'Orange_Haunglongbing_(Citrus_greening)',16:'Peach_Bacterial_spot',17:'Healthy_peach',18:'Pepper_bell_Bacterial_spot',19:'Pepper_bell_healthy',20:'Potato_Early_blight',21:'Healthy_potato',21:'Potato_Late_blight',22:'Healthy_Raspberry',23:'Healthy_soybean',24:'Squash_Powdery_mildew',25:'Healthy_Strawberry',26:'Strawberry_Leaf_scorch',27:'Tomato_Bacterial_spot',28:'Tomato_Early_blight',29:'Healthy_Tomato',30:' Tomato_Late_blight',31:'Tomato_Leaf_Mold',32:'Tomato_Septoria_leaf_spot',33:'Tomato_Spider_mites',34:'Tomato_Target_Spot',35:'Tomato_mosaic_virus',36:'Tomato_Yellow_Leaf_Curl_Virus'}  
    disease_result = disease_dict.get(disease_name[0], "Unknown Disease")

    st.write(f"The model has predicted that the leaf has **{disease_result}**.")

