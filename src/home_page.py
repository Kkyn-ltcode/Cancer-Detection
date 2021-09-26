import streamlit as st
from tensorflow.keras.preprocessing import image

def home_page():
    st.markdown("""
    <h1 style='text-align: center; color: black;'>
    Fast and accurate histopathology cancer 
    <br>
    images prediction in real-time
    </h1>""", unsafe_allow_html=True
    )

    st.markdown("""
    <p style='text-align: center;'>
    Cancer diagnosis with the model has been trained on 170,000 images 
    <br>
    and customizing beautiful bar chart with your data.
    <br>
    <br>
    <br>
    </p>
    """, unsafe_allow_html=True
    )

    upload_cols = st.columns(2)
    with upload_cols[0]:
        st.markdown("""
        <h2>
        <b>Upload your image<b>
        </h2>
        """, unsafe_allow_html=True
        )
        st.markdown("""
        <p>
        Your image(s) need to be histopathology cancer image(s) to classify more accurate with our model.
        <br>
        <br>
        You can search internet for more image like this one.
        </p>
        """, unsafe_allow_html=True
        )
    with upload_cols[1]:
        st.markdown("""
        <br>
        <br>
        """,
        unsafe_allow_html=True)
        img = image.load_img('test images/test_6.jpg', target_size=(200, 250))
        st.image(img)

    predict_plot_cols = st.columns(2)
    with predict_plot_cols[1]:
        st.markdown("""
        <h2>
        <br>
        <br>
        <b>Fast in prediction</b>
        </h2>
        """, unsafe_allow_html=True
        )

        st.markdown("""
        <p>
        Classify your image with a powerful neural network.
        <br>
        <br>
        <br>
        </p>
        """, unsafe_allow_html=True)
    with predict_plot_cols[0]:
        st.markdown("""
        <br>
        <br>
        <br>
        """,
        unsafe_allow_html=True)
        st.image('gif and bar charts/giphy3.gif')

    cols_top = st.columns(3)
    cols_bottom = st.columns(3)
    
    with cols_top[1]:
        st.markdown("""
        <br>
        <br>
        <br>
        """,
        unsafe_allow_html=True)
        st.image('gif and bar charts/plot1.png')
    with cols_top[2]:
        st.markdown("""
        <br>
        <br>
        <br>
        """,
        unsafe_allow_html=True)
        st.image('gif and bar charts/plot3.png')
    with cols_top[0]:
        st.markdown("""
        <h2>
        <br>
        <br>
        <b>Colorful bar chart</b>
        </h2>
        """, unsafe_allow_html=True
        )

        st.markdown("""
        <p>
        Design and paint your bar chart with many options supported by <b>matplotlib</b> library
        </p>
        """, unsafe_allow_html=True
        )

    with cols_bottom[1]:
        st.image('gif and bar charts/plot4.png')
    with cols_bottom[2]:
        st.image('gif and bar charts/plot2.png')