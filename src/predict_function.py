import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models

def show_default_bar_chart(data):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    data_cancer = data['Label'].value_counts().sort_index()
    ax.bar(data_cancer.index, data_cancer, width=0.55, edgecolor='#4a4a4a', color='#d4dddd', linewidth=0.7)

    step = max(data_cancer) // 6 + 1
    for i in data_cancer.index:
        ax.annotate(f"{data_cancer[i]}", xy=(i, data_cancer[i] + 0.1 * step), color='#4a4a4a', fontsize=12,
                    va = 'center', ha='center', fontweight='light', fontfamily='serif')


    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # ax.set_xticklabels(data_cancer.index, fontfamily='serif', fontsize=12)
    # ax.set_yticklabels(np.arange(0, max(data_cancer) + 1, step), fontfamily='serif', fontsize=12)
    fig.text(0.1, 0.95, 'Cancer Tumor Distribution', fontsize=15, fontweight='bold', fontfamily='serif')    
    ax.grid(axis='y', linestyle='-', alpha=0.4)
    st.write(fig)

def show_default_bar_horizontal_chart(data):
    fig, ax = plt.subplots(1,1, figsize=(12, 7))
    data_cancer = data['Label'].value_counts().sort_index()
    ax.barh(data_cancer.index, data_cancer, height=0.55, edgecolor='#4a4a4a', color='#d4dddd', linewidth=0.7)

    step = max(data_cancer) // 6 + 1
    for i in data_cancer.index:
        ax.annotate(f"{data_cancer[i]}", xy=(data_cancer[i] + 0.1 * step, i), color='#4a4a4a', fontsize=12,
                    va = 'center', ha='center',fontweight='light', fontfamily='serif')

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
 
    ax.set_yticklabels(data_cancer.index, fontfamily='serif', fontsize=12)
    ax.set_xticklabels(np.arange(0, max(data_cancer) + 1, step), fontfamily='serif', fontsize=12)
    fig.text(0.1, 0.95, 'Cancer Tumor Distribution', fontsize=15, fontweight='bold', fontfamily='serif')    
    ax.grid(axis='x', linestyle='-', alpha=0.4)
    st.write(fig)

def color_edit(color_options, labels, default_color):
    keys = [label.split(' ')[-1] for label in labels]  
    colors_list = []
    
    if 'set_num' not in  st.session_state:
        st.session_state.set_num = False

    with st.form('number of colors'):
        num_of_colors = []
        for i in range(3):
            if color_options[i] == 'Multiple':
                num_of_colors.append(st.slider(f'Number of {keys[i]} colors', 2, 10 ,2, 1))
            elif color_options[i] == 'Sync':
                num_of_colors.append(1)

        if st.form_submit_button('Apply changes'):
            st.session_state.set_num = True

    if st.session_state.set_num:
        with st.form('colors'):
            for i in range(3):
                if color_options[i] == 'Sync':
                    colors_list.append(st.color_picker(labels[i], default_color[i],))
                elif color_options[i] == 'Multiple':
                    sub_colors_list = []
                    index = 0
                    rows = 2 if num_of_colors[i] > 5 else 1
                    for j in range(rows):
                        cols = st.columns(5)
                        while index < num_of_colors[i]:
                            with cols[index % 5]:
                                sub_colors_list.append(st.color_picker(f'Color {keys[i][:-1]} {index + 1}', 
                                value=default_color[i], key=f'{keys[i]}_color_{index}'))
                            index += 1
                            if index - 1 == 4:
                                break
                    colors_list.append(sub_colors_list)
            submit = st.form_submit_button('Apply changes')

    return colors_list

def get_size_weight_family(label):
    font_weight = st.radio('Font weight', ['light', 'bold'], key=label + '0')

    font_size = st.slider('Font size', 1, 15, 12, 1, key=label + '1')

    font_family = st.selectbox('Font family', ['serif'], key=label + '2')
    return [font_size, font_weight, font_family]

def customize_bar_chart(data, bar, barh):
    data_cancer = data['Label'].value_counts().sort_index()
    step = max(data_cancer) // 6 + 1

    with st.form('Edit bar chart'):
        with st.expander('Figure size', expanded=False):
            width = st.number_input('Width', 5, 15, 12, 
            help='Enter a value from 5 to 15.')

            height = st.number_input('Height', 3, 9, 7,
            help='Enter a value from 3 to 9.')

        with st.expander('Bar edit', expanded=False):
            if barh:
                bar_height = st.slider('Bar height', -1., 1., 0.55,
                help='''The height of the bar, the bar will be adjacent if set to 1 
                or become straight line if set to 0, can be nagative.
                ''')
            elif bar:
                bar_width = st.slider('Bar width', -1., 1., 0.55,
                help='''The width of the bar, the bar will be adjacent if set to 1 
                or become straight line if set to 0, can be nagative.
                ''')

            align = st.radio('Alignment of the bar with *x* corrdinates',
            ['center', 'edge'], help='''Select **center** to center the base 
            of the bar, and **edge** to place the base to the left edge of the bar,
            select a negative **Bar width** to place the base to the right edge of the bar.
            ''')

            line_width = st.slider('Width of the bar edges', 0., 1., 0.7, 0.1, 
            help='If you dont want to draw line width, set value to 0')

            alpha = st.slider('Light intensity', 0., 1., 0.7, 0.1, 
            help="Change the intensity of bar color, choose a small value to blur the color")

        with st.expander('Annotation edit', expanded=False):
            top_cols = st.columns(3)
            with top_cols[0]:
                if barh:
                    text_width = st.number_input("Text's width", 0., 1. * step, 0.1 * step, 0.1 * step, 
                    help=f"Text's width is the different between text's base and right most edge of the bar, enter a number from 0 to {step}")
                elif bar:
                    text_height = st.number_input("Text's height", 0., 1. * step, 0.1 * step, 0.1 * step, 
                    help=f"Text's height is the different between text's base and top edge of the bar, enter a number from 0 to {step}")
            with top_cols[1]:
                ha = st.selectbox('Horizontal Alignment', ['center', 'right', 'left'], 
                help="Align the text's base horizontally")

            with top_cols[2]:
                va = st.selectbox('Vertical Alignment', ['center', 'top', 'bottom'], 
                help="Align the text's base vertically")

            bottom_cols = st.columns(3)
            with bottom_cols[0]:
                a_font_weight = st.radio('Font weight', ['light', 'bold'])

            with bottom_cols[1]:
                a_font_size = st.slider('Font size', 1, 15, 12, 1)

            with bottom_cols[2]:
                a_font_family = st.selectbox('Font family', ['serif'])

        submit = st.form_submit_button("Apply changes")

    with st.expander('Set axis', expanded=False):
        set_axis = st.radio('Set axis off', ['No', 'Yes'], help='Hide axis')
    if set_axis == 'No':
        with st.form('coordinate'):
            with st.expander('Visible', expanded=False):
                invisible_list = st.multiselect('Invisible frame list', ['top', 'right', 'bottom', 'left'], 
                ['top', 'right'], help='Hide figure frame')
            
            top_cols = st.columns(2)
            with top_cols[0]:
                with st.expander('X axis', expanded=False):
                    x_label = st.text_input('X label')
                    x_label_atb = get_size_weight_family('x_label')
            with top_cols[1]:
                with st.expander('Y axis', expanded=False):
                    y_label = st.text_input('Y label')
                    y_label_atb = get_size_weight_family('y_label')
                
            bottom_cols = st.columns(2)
            with bottom_cols[0]:
                with st.expander("X tick label", expanded=False):
                    x_tick_atb = get_size_weight_family('x_tick')
            with bottom_cols[1]:
                with st.expander("Y tick label", expanded=False):
                    y_tick_atb = get_size_weight_family('y_tick')

            axis_submit = st.form_submit_button('Apply changes')

    with st.expander('Title', expanded=False):
        with st.form('title'):
            cols = st.columns(2)
            with cols[0]:
                x_title_coor = st.slider('X coordinate', 0., 1., 0.1, 0.1)
            with cols[1]:
                y_title_coor = st.slider('Y coordinate', 0., 1., 0.95, 0.1)
            title = st.text_input('Title')
            title_abt = get_size_weight_family('title')
            title_submit = st.form_submit_button('Apply changes')

    with st.expander('Grid', expanded=False):
        grid = st.radio('Set grid', ['Yes', 'No'])
        if grid == 'Yes':
            with st.form('grid'):
                cols = st.columns(5)
                with cols[0]:
                    if barh:
                        axis = st.radio('Axis', ['x', 'y', 'both'], 
                        help='The axis to apply change for')
                    elif bar:
                        axis = st.radio('Axis', ['y', 'x', 'both'], 
                        help='The axis to apply change for')
                with cols[1]:
                    line_style = st.selectbox('Line style', ['-', '--'])
                with cols[2]:
                    grid_line_width = st.slider('Line width', 0.0, 2.0, 0.5, 0.1)
                with cols[3]:
                    line_color = st.color_picker('Line color')
                with cols[4]:
                    line_alpha = st.slider('Light intensity', 0., 1., 0.4, 0.1, 
                    help="Change the intensity of line color, choose a small value to blur the color")

                grid_submit = st.form_submit_button('Apply_changes')

    with st.expander('Background', expanded=False):
        with st.form('background'):
            cols = st.columns(2)
            with cols[0]:
                axis_bg = st.color_picker('Axis background color', '#FFFFFF')
                axis_alpha = st.slider("Axis's color intensity", 0., 1., 0.7, 0.1, 
                help="Change the intensity of axis background color, choose a small value to blur the color")
            with cols[1]:
                figure_color = st.color_picker('Figure background color', '#FFFFFF')
                figure_alpha = st.slider("Figure's color intensity", 0., 1., 0.7, 0.1, 
                help="Change the intensity of figure background color, choose a small value to blur the color")
                
            bg_submit = st.form_submit_button('Apply changes')

    if 'color_setted' not in st.session_state:
        st.session_state.color_setted = False

    color_list = ['#4a4a4a', '#d4dddd', ['#4a4a4a', '#4a4a4a']]
    with st.expander('Color mode', expanded=False):
        with st.form('color mode'):
            cols = st.columns(3)
            with cols[0]:
                face_color_option = st.radio('Face color option', ['Sync', 'Multiple'],
                help='Apply one or multiple color(s) for the bar faces')
            with cols[1]:
                edge_color_option = st.radio('Edge color option', ['Sync', 'Multiple'],
                help='Apply one or multiple color(s) for the bar edges')
            with cols[2]:
                annotation_color_option = st.radio('Annotation color option', ['Sync', 'Multiple'],
                help='Apply one or multiple color(s) for the annotation')

            if st.form_submit_button('Apply change'):
                st.session_state.color_setted = True

    if st.session_state.color_setted:
        labels = ['The colors of the bar edges', 'The colors of the bar faces', 'The colors of the annotations']
        color_options = [edge_color_option, face_color_option, annotation_color_option]
        color_defaults = ['#4a4a4a', '#d4dddd', '#4a4a4a']
        if color_options.count('Sync') != 3:
            with st.expander('Color setting'):
                color_list = color_edit(color_options, labels, color_defaults)
        else:
            with st.expander('Color setting'):
                with st.form('color'):
                    cols = st.columns(3)
                    with cols[0]:
                        edge_color = st.color_picker('The colors of the bar edges', '#4a4a4a')
                    with cols[1]:
                        face_color = st.color_picker('The colors of the bar faces', '#d4dddd')
                    with cols[2]:
                        annotation_color = st.color_picker('The colors of the annotations', '#4a4a4a')
                    color_submit = st.form_submit_button('Apply changes')
            color_list = [edge_color, face_color, [annotation_color, annotation_color]]

    if st.button('Plot'):
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
 
        if barh:
            ax.barh(data_cancer.index, data_cancer, height=bar_height, edgecolor=color_list[0], color=color_list[1], 
            linewidth=line_width, alpha=alpha, align=align)
        elif bar:
            ax.bar(data_cancer.index, data_cancer, width=bar_width, edgecolor=color_list[0], color=color_list[1], 
            linewidth=line_width, align=align, alpha=alpha)

        index = 0
        for i in data_cancer.index:
            if barh:
                ax.annotate(f"{data_cancer[i]}", xy=(data_cancer[i] + text_width, i), color=color_list[2][index],
                            va=va, ha=ha, fontweight=a_font_weight, fontfamily=a_font_family, fontsize=a_font_size)
            elif bar:
                ax.annotate(f"{data_cancer[i]}", xy=(i, data_cancer[i] + text_height), color=color_list[2][index],
                            va=va, ha=ha, fontweight=a_font_weight, fontfamily=a_font_family, fontsize=a_font_size)
            index += 1
        if set_axis == 'No':
            for s in invisible_list:
                ax.spines[s].set_visible(False)

            if barh:
                ax.set_yticklabels(data_cancer.index, fontfamily=y_tick_atb[2], fontsize=y_tick_atb[0], fontweight=y_tick_atb[1])
                ax.set_xticklabels(np.arange(0, max(data_cancer) + 1, step), fontfamily=x_tick_atb[2], fontsize=x_tick_atb[0], 
                fontweight=x_tick_atb[1])
            elif bar:
                ax.set_xticklabels(data_cancer.index, fontfamily=x_tick_atb[2], fontsize=x_tick_atb[0], fontweight=x_tick_atb[1])
                ax.set_yticklabels(np.arange(0, max(data_cancer) + 1, step), fontfamily=y_tick_atb[2], fontsize=y_tick_atb[0], 
                fontweight=y_tick_atb[1])

            ax.set_xlabel(x_label, fontfamily=x_label_atb[2], fontsize=x_label_atb[0], fontweight=x_label_atb[1])
            ax.set_ylabel(y_label, fontfamily=y_label_atb[2], fontsize=y_label_atb[0], fontweight=y_label_atb[1])

        fig.text(x_title_coor, y_title_coor, title, fontsize=title_abt[0], fontweight=title_abt[1], fontfamily=title_abt[2])    

        if grid == 'Yes':   
            ax.grid(axis=axis, linestyle=line_style, alpha=line_alpha, linewidth=grid_line_width, color=line_color)

        if set_axis == 'Yes':   
            ax.set_axis_off()

        ax.patch.set_facecolor(axis_bg)
        ax.patch.set_alpha(axis_alpha)
        fig.patch.set_facecolor(figure_color)
        fig.patch.set_alpha(figure_alpha)

        st.pyplot(fig)

def show_statistic(data):
    if 'statistic_option' not in st.session_state:
        st.session_state.statistic_option = False
    with st.expander('Statistic option', expanded=True):
        with st.form('bar chart'):
            cols = st.columns(3)
            with cols[0]:
                plot_type = st.radio('Plot type', ['Tumor distribution', 'Probability distribution (comming soon)'])
            with cols[1]:
                bar_type = st.radio('Bar type', ['Bar', 'Bar horizontal'])
            with cols[2]:
                setting = st.radio('Setting', ['Default', 'Custom'])

            if st.form_submit_button('Apply changes'):
                st.session_state.statistic_option = True
            
    if st.session_state.statistic_option:
        if setting == 'Default':
            if plot_type == 'Tumor distribution':
                if bar_type == 'Bar':
                    show_default_bar_chart(data)
                elif bar_type == 'Bar horizontal':
                    show_default_bar_horizontal_chart(data)
            # elif plot_type == 'Probability distribution':
            #     # if bar_type == 'Bar':
            #     #     pro_bar_chart(data)
            #     pass
            
        elif setting == 'Custom':
            if plot_type == 'Tumor distribution':
                if bar_type == 'Bar':
                    customize_bar_chart(data, True, False)
                elif bar_type == 'Bar horizontal':
                    customize_bar_chart(data, False, True)

def predicting(imgs, file_df):
    with st.spinner('Predicting...'):
        st.image('gif and bar charts/giphy2.gif')
        imgs = [image.img_to_array(img) / 255 for img in imgs]
        imgs = np.array(imgs)
        model = models.load_model('model/cancer_detection.h5')
        proba = model.predict(imgs)
        results = [i[0] for i in proba]
        labels = ['Benign' if result < 0.5 else 'Malignant' for result in proba]
        prediction =  pd.DataFrame(zip(file_df['Image name'], results, labels), 
        columns=['Image name', 'Score', 'Label'])
    st.success('Done!')
    return prediction

def predict_function(imgs, file_df, predicted):
    imgs = [img.resize((96, 96)) for img in imgs]
    if 'state' not in st.session_state:
        st.session_state.state = pd.DataFrame()
    if 'statistic_mode' not in st.session_state:
        st.session_state.statistic_mode = False

    with st.expander('Learn more about these options', expanded=True):
        st.markdown("""
        <p>
        <b>Model architecture</b> visualize the deep leanrning neural network.
        </p>
        <p>
        <b>File infomation</b> display your image's elements such as <i>name</i>, <i>type</i>, <i>size</i>, <i>shape</i> of image
        <br>
        in height-width-channels format.
        <br>
        </p>
        <p>
        <b>Predict</b> is just... make prediction.
        </p>
        <p>
        <b>View result</b> shows the prediction of your image(s).
        <br>
        There are two possible labels, <b>Benign</b> and <b>Malignant</b>. Most benign tumors are not harmful, 
        and they are unlikely to affect other parts of the body. Malignant tumors are cancerous. 
        They can grow quickly and spread to other parts of the body so they can become life threatening.
        </p>
        <p>
        <b>View statistic</b> and draw bar chart base on your result. You can customize your chart with many colors and other options.
        </p>
        """, unsafe_allow_html=True)

    with st.expander('Select option', expanded=True):
        cols = st.columns([1.4, 1.2, 0.8, 1, 1.2])
        with cols[0]:
            model_architecture = st.button('Model architecture', help='Visualize the model')
        with cols[1]:
            file_info = st.button('File infomation')
        with cols[2]:
            predict = st.button('Predict', help='Feed the uploaded images to the model')
        with cols[3]:
            result = st.button('View result', help='Show the result of prediction')
        with cols[4]:
            statistic = st.button('View statistic', help='Plot your result with bar chart')
        
        if model_architecture:
            st.image('model/model_plot.png')
        
        if file_info:
            st.write(file_df)

        if predict:
            predicted = True
            st.session_state.state = predicting(imgs, file_df)

        if result:
            if not st.session_state.state.empty:
                st.write(st.session_state.state)
            else:
                st.warning("You Haven't Made Any Prediction Yet")

        if statistic:
            if not st.session_state.state.empty:
                st.session_state.statistic_mode = True
            else:
                st.warning("You Haven't Made Any Prediction Yet")

    if st.session_state.statistic_mode:
        show_statistic(st.session_state.state)
            
    return predicted, st.session_state.state