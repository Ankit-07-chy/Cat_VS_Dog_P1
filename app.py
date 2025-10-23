import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


st.set_page_config(
    page_title='Dog Vs Cat Classifier !',
    layout='centered'
)

model = load_model('Cat_dog.keras')
st.header('Dog vs Cat Classifier')


uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    image = tf.keras.utils.load_img(uploaded_file, target_size=(256, 256))
    img_arr = tf.keras.utils.img_to_array(image)
    img_arr = img_arr / 255.0
  
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img_arr, caption='Processed Image', use_container_width=True)

    img_bat = tf.expand_dims(img_arr, 0)  #creating batch for prediction

    # Make prediction

    prediction = model.predict(img_bat)

    score = prediction[0][0]
    if score > 0.5:
        result = "Dog"
        
        confidence = score * 100
    else:
        result = "Cat"
        confidence = (1 - score) * 100
    
    #st.write(result,confidence)
    st.title(f"The prediction for the image is {result} with a confidence of {confidence:.2f}%")


