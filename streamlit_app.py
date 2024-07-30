import os
import io
import json
import pandas as pd
import numpy as np
from itertools import combinations
# from scipy.sparse import dok_matrix
# from scipy.spatial import distance
# from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import glob

from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
import plotly.colors as colors

import viz_functions as viz

import streamlit as st
# from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from datetime import datetime

# @st.experimental_memo
st.set_page_config(layout="wide")

def load_pickle_slices(base_filename):
    chunk_files = sorted(glob.glob(f"{base_filename}_p*.pkl"))
    
    # Combine the chunks back into a single byte stream
    byte_stream = io.BytesIO()
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as file:
            byte_stream.write(file.read())
    
    byte_stream.seek(0)
    obj = pickle.load(byte_stream)
    
    return obj


def main():
    with st.sidebar:
        st.subheader('Volumetric Tracing with Super-Resolution Microscopy')
        st.write('M.Unal, UT MD Anderson Cancer Center (2023)')
        st.write('''
                This presentation showcases a collaborative effort between Nir's Lab at UTMB and Akdemir Lab at UTMDACC, 
                focusing on the intricate analysis of genome structures using cutting-edge super-resolution microscopy. 
                The high-resolution images, obtained through state-of-the-art microscopy techniques in Nir's Lab, provide 
                unprecedented insights into cellular processes. These images capture cellular details at the nanoscale, 
                offering a glimpse into the complexity of cellular structures. Cell lines, meticulously produced in Akdemir Lab, 
                serve as the foundation for this comprehensive analysis. The data analysis presented here, a crucial aspect of this collaborative 
                endeavor, unveils hidden patterns and structures within the microscopic images. In the latest version of the analysis pipeline, 
                k-d tree algorithm is used to match high resolution (backstreet) data to low resolution data (mainstreet).
                ''')

        st.write('*Click on the dataset in the dropdown menu.*')


        # dataset = st.selectbox("Dataset", ['Trace1_Location2_Cell1, precision x,y,z > 0',
        #                                     'Trace1_Location2_Cell1, precision x,y,z > 12.5',
        #                                    'Trace1_Location4_Cell1, precision x,y,z > 0',
        #                                    'Trace1_Location4_Cell1, precision x,y,z > 10'
        #                                    ], index=0)
        
        dataset = [

                    "Set1_Location2_Cell1",
                    "Set1_Location4_Cell1",
                    "Set1_Location4_Cell2",
                    "Set1_Location5_Cell1",
                    "Set1_Location5_Cell2",
                    "Set3_Location2_Cell1",
                    "Set3_Location2_Cell3",
                    "Set3_Location6_Cell2",
                    "Set3_Location7_Cell2",
                    "Set3_Location8_Cell1",
                    "Set3_Location8_Cell2",
                    "Set3_Location8_Cell3",
                    "Set3_Location9_Cell1",
                    "Set3_Location9_Cell2",

                    "PhChr_Set1_Location2_Cell1_Chr9",
                    "PhChr_Set1_Location2_Cell2_Chr9",
                    "PhChr_Set1_Location2_Cell3_Chr9",
                    "PhChr_Set1_Location2_Cell4_Chr9",
                    "PhChr_Set1_Location3_Cell1_Chr9",
                    "PhChr_Set1_Location3_Cell2_Chr9",
                    "PhChr_Set1_Location3_Cell3_Chr9",
                   ]
        
        selected_dataset = st.selectbox("Choose dataset", dataset)

        # # Define options for the second selectbox based on the selected location
        # options_mapping = {
        #     # 'Set1_Location2_Cell1': ['precision x,y,z > 0'],
        #     'Set1_Location4_Cell1': ['precision x,y,z > 0'],
        #     'Set3_Location8_Cell3': ['precision x,y,z > 0'],
        #     'Set3_Location9_Cell2': ['precision x,y,z > 0'],
        #     # 'PhChr_Set1_Location1_Cell1_Chr9': ['precision x,y,z > 0'],
        #     # 'PhChr_Set1_Location1_Cell1_Chr22': ['precision x,y,z > 0'],
        #     'PhChr_Set1_Location2_Cell1_Chr9': ['precision x,y,z > 0'],
        #     # 'PhChr_Set1_Location2_Cell1_Chr22': ['precision x,y,z > 0'],
        # }
        # selected_option = st.selectbox("Choose precision threshold", options_mapping[selected_dataset])

        exp = selected_dataset
        # cut = selected_option.split(" ")[-1]
        cut = 0

        if exp.split("_")[0] == 'PhChr':
            if exp.split("_")[-1] == 'Chr22':
                MAINSTREET_TP_RANGE= (7,14) 
                BACKSTREET_TP_RANGE= (31,35)
            else: #Chr9
                MAINSTREET_TP_RANGE= (0,15)
                BACKSTREET_TP_RANGE= (16,20)
        else:
            MAINSTREET_TP_RANGE = (0,19) # (first,last)
            BACKSTREET_TP_RANGE = (20,24) # (first,last)

        base_path = f"saves_kdtree/{exp}/precision_{cut}"
    
        file_patterns = ['traces', 'analysis_results_dict', 'resolution_results_dict']

        traces = None
        analysis_results_dict = None
        resolution_results_dict = None

        # Loop through file patterns and merge pickle files
        for pattern in file_patterns:
            base_filename = os.path.join(base_path, pattern)
            if pattern == "traces":
                traces = load_pickle_slices(base_filename)
            elif pattern == "analysis_results_dict":
                analysis_results_dict = load_pickle_slices(base_filename)
            elif pattern == "resolution_results_dict":
                resolution_results_dict = load_pickle_slices(base_filename)


        ## ToC
        st.sidebar.markdown("# Table of Contents")
        st.sidebar.markdown("1. [Analysis](#analysis)")
        st.sidebar.markdown("   1.1. [3D scatter plots of blinking events](#scatter-plots)")
        st.sidebar.markdown("   1.2. [3D scatter plots of backstreet assignments over mainstreet data](#scatter-20kb)")
        st.sidebar.markdown("   1.3. [Backstreet assignments](#bs-assignment)")
        st.sidebar.markdown("   1.4. [Predicted vs. random assigments of backstreet blinking eventss](#algorithm-vs-random)")

        ### CONTACT MAPSSSSSSSS

        st.sidebar.markdown("2. [Appendix](#appendix)")
        st.sidebar.markdown("   2.1. [A1. Pairwise distances (Moat) histograms](#com-radius)")

        #### Report pannel
        st.subheader('Bug Report/Recommendations')
        with st.form("bug_report"):
            st.write("Feedback Form")
            url = "https://docs.google.com/spreadsheets/d/18O4g3OX4XlS41yMsV_kMLgG1li8gvVJyCTaXJZ6kF6k/edit#gid=0"
            st.write("[Link to the feedback list](%s)"%url)
            cols = st.columns((1, 1))
            author = cols[0].text_input("Report author:")
            fb_type = cols[1].selectbox(
                "Report type:", ["Bug", "Recommandation"], index=1
            )
            comment = st.text_area("Comment:")
            cols = st.columns(2)
            bug_severity = cols[1].slider("Severity/Importance:", 1, 5, 2)
            submitted = st.form_submit_button("Submit")

            if submitted:
                # Automatically get the current date
                date = datetime.now().date()
                data = [[date, author, fb_type, bug_severity, comment]]
                # Your spreadsheet ID and range
                spreadsheet_id = '18O4g3OX4XlS41yMsV_kMLgG1li8gvVJyCTaXJZ6kF6k'
                range_name = 'Sheet1!A:E'  # Make sure to adjust the range according to your Google Sheet's structure
                viz.append_data_to_sheet(data, spreadsheet_id, range_name)
                st.success("Bug reported successfully!")


    st.header("Analysis")
    st.markdown("<a id='analysis'></a>", unsafe_allow_html=True)
    st.subheader("1- 3D scatter plots of blinking events")
    st.markdown("<a id='scatter-plots'></a>", unsafe_allow_html=True)
    ## 3D plots 
    col1,col2 = st.columns([2,2])
    with col1:
        ## TRACE 1
        color_set = "Light24"
        fig11 = viz.plotly_3D(traces['tr_1'],color_set,f"{exp} - {cut} - Trace 1 data")
        st.plotly_chart(fig11, use_container_width=True)
    if "tr_2" in traces:
        with col2:
            ## TRACE 2
            color_set = "Light24"
            fig12 = viz.plotly_3D(traces['tr_2'],color_set,f"{exp} - {cut} - Trace 2 data")
            st.plotly_chart(fig12, use_container_width=True)
    else:
        with col2:
            st.image('trilobite-fossils.jpg', caption='Trace 2 not exist')

    ## 3D scatter plots of 20 kb resolution data after backstreet assignments
    st.subheader("2- 3D scatter plots of backstreet assignments over mainstreet data")
    st.markdown("<a id='scatter-20kb'></a>", unsafe_allow_html=True)
    st.markdown('*Hover over data to see new time-point values \"New Time Point\".*')
    col1,col2 = st.columns([2,2])
    with col1:
        ## TRACE 1 
        fig21 = viz.plot_3d_time_series_with_dropdown(traces["tr_1"][traces["tr_1"]['time-point']<BACKSTREET_TP_RANGE[0]],
                                                       analysis_results_dict['match_results'][0]["tr_1"])
        st.plotly_chart(fig21, use_container_width=True)
    if "tr_2" in traces:
        with col2:
            ## TRACE 2
            fig22 = viz.plot_3d_time_series_with_dropdown(traces["tr_2"][traces["tr_2"]['time-point']<BACKSTREET_TP_RANGE[0]],
                                                        analysis_results_dict['match_results'][0]["tr_2"])
            st.plotly_chart(fig22, use_container_width=True)
    else:
        with col2:
            st.image('trilobite-fossils.jpg', caption='Trace 2 not exist')

    ## Backst Assignments
    st.subheader("3- Backstreet assignments")
    st.markdown("<a id='bs-assignment'></a>", unsafe_allow_html=True)
    col1,col2 = st.columns([2,2])
    with col1:
        ## TRACE 1
        fig311 = viz.plotly_backst_distibutions(analysis_results_dict['match_results'][0]["tr_1"],traces["tr_1"],"tr1",MAINSTREET_TP_RANGE)
        st.plotly_chart(fig311, use_container_width=True)
        
        fig312 = viz.plotly_Sankey_diagram(analysis_results_dict['match_results'][0]["tr_1"],"tr1")
        st.plotly_chart(fig312, use_container_width=True)
        
        norm_flag1 = st.selectbox("Normalize", [True, False], index=0, key='norm1')
        fig313 = viz.backst_dist(analysis_results_dict['match_results'][0]["tr_1"],norm_flag1)
        st.plotly_chart(fig313, use_container_width=True)
    if "tr_2" in traces:
        with col2:
            ## TRACE 2
            fig321 = viz.plotly_backst_distibutions(analysis_results_dict['match_results'][0]["tr_2"],traces["tr_2"],"tr2",MAINSTREET_TP_RANGE)
            st.plotly_chart(fig321, use_container_width=True)
            
            fig322 = viz.plotly_Sankey_diagram(analysis_results_dict['match_results'][0]["tr_2"],"tr2")
            st.plotly_chart(fig322, use_container_width=True)
            
            norm_flag2 = st.selectbox("Normalize", [True, False], index=0, key='norm2')
            fig323 = viz.backst_dist(analysis_results_dict['match_results'][0]["tr_2"],norm_flag2)
            st.plotly_chart(fig323, use_container_width=True)
    else:
        with col2:
            st.image('trilobite-fossils.jpg', caption='Trace 2 not exist')

    ## Backst Assignments vs Random Assignments
    st.subheader("4- Predicted vs. random assigments of backstreet blinking events")
    st.markdown("<a id='algorithm-vs-random'></a>", unsafe_allow_html=True)

    col1,col2 = st.columns([2,2])
    with col1:
        ## TRACE 1      
        fig411 = viz.plotly_backst_distibutions_with_randoms(analysis_results_dict['match_results'][0]["tr_1"],
                                                            traces['tr_1'],
                                                            analysis_results_dict['random_match_results']['tr_1'],
                                                            "tr1",MAINSTREET_TP_RANGE)
        st.plotly_chart(fig411, use_container_width=True)
    
        distances_tr1,distances_random_tr1 = viz.calc_distances(traces['tr_1'],
                                                            analysis_results_dict['match_results'][0]['tr_1'],
                                                            analysis_results_dict['random_match_results']['tr_1'])
        fig412 = viz.plotly_random_vs_prediction(distances_tr1,distances_random_tr1,"tr1",MAINSTREET_TP_RANGE)
        st.plotly_chart(fig412, use_container_width=True)

        fig413 = viz.plotly_box_plot(distances_tr1,distances_random_tr1,"tr1")
        st.plotly_chart(fig413, use_container_width=True)
    
        st.markdown(f":blue[Mann-Whitney U test *p-value*: {viz.non_parametric_tests(distances_tr1,distances_random_tr1)}]")

    if "tr_2" in traces:
        with col2:
            ## TRACE 2
            fig421 = viz.plotly_backst_distibutions_with_randoms(analysis_results_dict['match_results'][0]["tr_2"],
                                                                traces['tr_2'],
                                                                analysis_results_dict['random_match_results']['tr_2'],
                                                                "tr2",MAINSTREET_TP_RANGE)
            st.plotly_chart(fig421, use_container_width=True)
        
            distances_tr2,distances_random_tr2 = viz.calc_distances(traces['tr_2'],
                                                                analysis_results_dict['match_results'][0]['tr_2'],
                                                                analysis_results_dict['random_match_results']['tr_2'])
            fig422 = viz.plotly_random_vs_prediction(distances_tr2,distances_random_tr2,"tr2",MAINSTREET_TP_RANGE)
            st.plotly_chart(fig422, use_container_width=True)

            fig423 = viz.plotly_box_plot(distances_tr2,distances_random_tr2,"tr2")
            st.plotly_chart(fig423, use_container_width=True)
        
            st.markdown(f":blue[Mann-Whitney U test *p-value*: {viz.non_parametric_tests(distances_tr2,distances_random_tr2)}]")
    else:
        with col2:
            st.image('trilobite-fossils.jpg', caption='Trace 2 not exist')
    
    ## Initialize session state for histograms
    if 'histogram1' not in st.session_state:
        st.session_state['histogram1'] = None
    if 'histogram2' not in st.session_state:
        st.session_state['histogram2'] = None

    ## Appendix
    st.header("Appendix")
    st.subheader("A1- Pairwise distances histograms")
    st.markdown(f"*Depending on the dataset size, this may take a while to generate. That's why it is optional. Don't forget to push generate again after changing the dataset.* ")

    ## Moat histograms
    col1, col2 = st.columns([2, 2])

    with col1:
        if st.button('Generate pairwise distances histogram for trace 1'):
            st.session_state['histogram1'] = viz.compute_histogram_per_tp(traces['tr_1'][traces['tr_1']['time-point'] < BACKSTREET_TP_RANGE[0]])
        if st.session_state['histogram1'] is not None:
            figa11 = viz.plotly_pwd_histogram_with_dropdown(st.session_state['histogram1'])
            st.plotly_chart(figa11, use_container_width=True)

    if "tr_2" in traces:
        with col2:
            if st.button('Generate pairwise distances histogram for trace 2'):
                st.session_state['histogram2'] = viz.compute_histogram_per_tp(traces['tr_2'][traces['tr_2']['time-point'] < BACKSTREET_TP_RANGE[0]])
            if st.session_state['histogram2'] is not None:
                figa12 = viz.plotly_pwd_histogram_with_dropdown(st.session_state['histogram2'])
                st.plotly_chart(figa12, use_container_width=True)
    else:
        with col2:
            st.image('trilobite-fossils.jpg', caption='Trace 2 not exist')


if __name__ == '__main__':
    main()