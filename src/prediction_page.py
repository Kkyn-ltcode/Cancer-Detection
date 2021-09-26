import streamlit as st
from PIL import Image
import pandas as pd
from tensorflow.keras.preprocessing import image

from plot_images_function import plot_images_function
from predict_function import predict_function

def prepared_images(files):
    columns = ['Image name', 'Image size', 'Image type']
    files_info = [[file.name, file.size, file.type] for file in files]
    imgs = [Image.open(file) for file in files]
    img_shape = [image.img_to_array(img).shape for img in imgs]

    img_shape_str = pd.Series([f'{shape[0]}x{shape[1]}x{shape[2]}' 
    for shape in img_shape], name='Image shape')

    file_df = pd.DataFrame(files_info, columns=columns) 
    file_df = pd.concat([file_df, img_shape_str], axis=1)

    return imgs, file_df

def show_functions(files):
    imgs, file_df = prepared_images(files)

    if 'predicted' not in st.session_state:
        st.session_state.predicted = False
    if 'prediction' not in st.session_state:
        st.session_state.prediction = pd.DataFrame()

    sub_menu = ['Predict images', 'Plot images']
    user_choice = st.sidebar.selectbox('Select function', sub_menu)

    if user_choice == 'Predict images':
        st.session_state.predicted, st.session_state.prediction = predict_function(imgs, 
        file_df, st.session_state.predicted)

    elif user_choice == 'Plot images':
        plot_images_function(imgs, st.session_state.predicted, 
        st.session_state.prediction, file_df)

def prediction_page():
    files = ''
    with st.sidebar.expander('Upload file', expanded=True):
        files = st.file_uploader('Image here (You can upload multiple images)', 
        accept_multiple_files=True, type=['jpg', 'png', 'jpeg', 'tif'])

    if files:
            show_functions(files)
    else:
        st.markdown("""
        <h1>
        Upload your image to get started
        </h1>
        """, unsafe_allow_html=True)