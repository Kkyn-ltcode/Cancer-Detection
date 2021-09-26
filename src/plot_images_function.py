import streamlit as st
import numpy as np
import time

def plot_image(img_list, num_of_img, img_per_row, num_of_row, img_name):
    img_remaining = num_of_img
    for row in range(num_of_row):
        if (img_remaining < img_per_row):
            st.image(img_list[row * img_per_row: row * img_per_row + img_remaining],
            caption=img_name[row * img_per_row: row * img_per_row + img_remaining])
        else:
            st.image(img_list[row * img_per_row: row * img_per_row + img_per_row],
            caption=img_name[row * img_per_row: row * img_per_row + img_per_row])
        img_remaining -= img_per_row

def get_img_solution():
    with st.expander('Size option', expanded=True):
        width = st.number_input('Width', 120, 200, 120, 10, 
        help='Enter a number from 120 to 200')
        
        height = st.number_input('Height', 120, 200, 120, 10, 
        help='Enter a number from 120 to 200')
        return f'{int(width)}x{int(height)}'

def get_num_of_image(imgs):
    with st.expander('Number of image', expanded=True):
        return st.slider('Slide the bar', 1, len(imgs), len(imgs), 
        help='This option will relate to **Plot mode** so be careful')
            
def get_plot_mode():
    with st.expander('Plot mode', expanded=True):
        user_choice = st.radio('Select mode', ['Full', 'Random', 'Custom'], 
        help='''
        This option will change the mode of plot related to **Number of image**, both **Random** and
        **Custom** are recomment for ***none-full-selected-image***, meanwhile **Full** mode will only
        be selected if your **Number of image** feature is maximum.
        ''', )
        if user_choice == 'Random':
            return 'Random'
        elif user_choice == 'Custom':
            return 'Custom'
        elif user_choice == 'Full':
            return 'Full'

def get_row(num_of_img):
    with st.expander('Row edit', expanded=True):
        img_per_row = st.slider('Column', 1, num_of_img, 
        help='''
        This option will also change the **Number of row** bases on **Image per row** 
        (or **Column** if you ask) and **Number of image**''')
        num_of_row = int((num_of_img - 1) / img_per_row + 1)
        return img_per_row, num_of_row

def get_image_list( num_of_img, name_map_image, plot_mode, img_name):
    full_image = [value for key, value in name_map_image.items()]
    full_name = [key for key, value in name_map_image.items()]

    if plot_mode == 'Random':
        random_name = np.random.choice(img_name, num_of_img, replace=False)
        return [name_map_image[key] for key in random_name]
    elif plot_mode == 'Custom':
        selected_name =  st.multiselect('Select image', full_name, full_name[:num_of_img])
        return [name_map_image[key] for key in selected_name]
    elif plot_mode == 'Full':
        return full_image

def prepare_plot(imgs, img_size, img_name, num_of_img, plot_mode):
    width = int(img_size.split('x')[0])
    height = int(img_size.split('x')[1])
    imgs = [img.resize((width, height)) for img in imgs]
    name_map_img = {name: img for name, img in zip(img_name, imgs)}
    img_list = get_image_list(num_of_img, name_map_img, plot_mode, img_name)
    return img_list

def display_features(img_size, num_of_img, num_of_row, img_per_row, plot_mode):
    st.text_area('Features infomation', 
    f'''
    Image size: {img_size}
    Number of image: {num_of_img}
    Number of row: {num_of_row}
    Image per row: {img_per_row}
    Plot mode: {plot_mode}
    ''', height=12)

def get_img_name(prediction, file_df, include_label):
    if include_label:
        return [f"{img_name.split('.')[0]}-{label}"
        for img_name, label in zip(prediction['Image name'], prediction['Label'])]
    else:
        return [img_name.split('.')[0] for img_name in file_df['Image name']]

def get_default_values(imgs):
    default_dict = {
        'num_of_row': len(imgs),
        'num_of_image': len(imgs),
        'img_per_row': 1,
        'plot_mode': 'Full',
        'img_size': '96x96'    
    }
    return default_dict

def get_include_label(display_checkbox):
    if display_checkbox:
        return st.checkbox('Include prediction label(s)', 
        help='Name of image will be added with the trained label of that image')
    else:
        return False

def plot_images_function(imgs, display_checkbox, prediction, file_df):
    st.markdown("""
    <h2>
    Plot your uploaded image(s).
    <br>
    <br>
    </h2>
    """, unsafe_allow_html=True)

    default_option = get_default_values(imgs)
    cols_out = st.columns(2)
    with cols_out[0]:
        select_option = st.radio('Select option', ['Default', 'Custom'], 
        help='Choose **Custom** if you want to customize the elements of your image and plot, otherwise choose **Default**')
        include_label = get_include_label(display_checkbox)

    img_name = get_img_name(prediction, file_df, include_label)

    if select_option == 'Default':
        num_of_img = default_option['num_of_image']
        num_of_row = default_option['num_of_row']
        img_per_row = default_option['img_per_row']
        plot_mode = default_option['plot_mode']
        img_size = default_option['img_size']
        img_list = prepare_plot(imgs, img_size, img_name, num_of_img, plot_mode)
        with cols_out[1]:
            display_features(img_size, num_of_img, num_of_row, img_per_row, plot_mode)
        if st.button('Plot'):
            plot_image(img_list, num_of_img, img_per_row, num_of_row, img_name)

    elif select_option == 'Custom':
        img_size = get_img_solution()
        cols = st.columns(2)
        with cols[0]:
            num_of_img = get_num_of_image(imgs)
        with cols[1]:
            img_per_row, num_of_row = get_row(num_of_img) 

        plot_mode = get_plot_mode()
        img_list = prepare_plot(imgs, img_size, img_name, num_of_img, plot_mode)

        with cols_out[1]:
            display_features(img_size, num_of_img, num_of_row, img_per_row, plot_mode)

        if num_of_img != len(imgs):
            if plot_mode == 'Full':
                warning = '''
                You may want to check your **Number of image** feature before select this mode, 
                but if you know what you are doing, just ignore this message!
                '''
                with st.spinner(warning):
                    time.sleep(5)
                with st.spinner('Checking...'):
                    time.sleep(2)
                st.error('Please select the right options!')
                
            elif plot_mode in ['Random', 'Custom']:
                if len(img_list) != num_of_img:
                    st.error('Please select the right options!')
                else:
                    if st.button('Plot'):
                        plot_image(img_list, num_of_img, img_per_row, num_of_row, img_name)
        else:
            if st.button('Plot'):
                plot_image(img_list, num_of_img, img_per_row, num_of_row, img_name)