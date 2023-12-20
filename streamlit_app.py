import os
import json
import pandas as pd
import numpy as np
from itertools import combinations
# from scipy.sparse import dok_matrix
# from scipy.spatial import distance
# from scipy.spatial.distance import cdist

import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
import plotly.colors as colors

import viz_functions as viz

import streamlit as st
# from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

# @st.experimental_memo
st.set_page_config(layout="wide")

def main():
    with st.sidebar:
        st.subheader('Volumetric Tracing with Super-Resolution Microscopy')
        st.write('M.Unal, UT MD Anderson Cancer Center (2023)')
        st.write('''
                This presentation showcases a collaborative effort between Nir's Lab at UTMB and Akdemir Lab at UTMDACC, 
                focusing on the intricate analysis of genome structures using cutting-edge super-resolution microscopy. 
                The high-resolution images, obtained through state-of-the-art microscopy techniques in Nir's Lab, provide 
                unprecedented insights into cellular processes. These images capture cellular details at the nanoscale, 
                offering a glimpse into the complexity of cellular structures.Cell lines, meticulously produced in Akdemir Lab, 
                serve as the foundation for this comprehensive analysis. The data analysis, a crucial aspect of this collaborative 
                endeavor, unveils hidden patterns and structures within the microscopic images.            
                ''')

        st.write('*Click on the dataset in the dropdown menu.*')


        # dataset = st.selectbox("Dataset", ['Trace1_Location2_Cell1, precision x,y,z > 0',
        #                                     'Trace1_Location2_Cell1, precision x,y,z > 12.5',
        #                                    'Trace1_Location4_Cell1, precision x,y,z > 0',
        #                                    'Trace1_Location4_Cell1, precision x,y,z > 10'
        #                                    ], index=0)
        
        dataset = ['Trace1_Location2_Cell1', 'Trace1_Location4_Cell1']
        selected_dataset = st.selectbox("Choose dataset", dataset)

        # Define options for the second selectbox based on the selected location
        options_mapping = {
            'Trace1_Location2_Cell1': ['precision x,y,z > 0', 'precision x,y,z > 12.5'],
            'Trace1_Location4_Cell1': ['precision x,y,z > 0', 'precision x,y,z > 10']
        }

        selected_option = st.selectbox("Choose Precision Option", options_mapping[selected_dataset])

        exp = selected_dataset
        cut = selected_option.split(" ")[-1]

        files = [#"df_entire_cl1","df_entire_cl2",
                "df_entire_cl1_precise","df_entire_cl2_precise",
                #"df_entire_concat",
                "df_cl1_com_R100", "df_cl2_com_R100", 
                "shortest_paths_cl1_R100", "shortest_paths_cl2_R100",
                "shortest_path_coordinates_cl1_R100", "shortest_path_coordinates_cl2_R100", 
                "arr_mainstreet_cl1_R100", "arr_mainstreet_cl2_R100", 
                #"df_tp0_concat",
                "df_cl1_com_R200", "df_cl2_com_R200", 
                "shortest_paths_cl1_R200", "shortest_paths_cl2_R200", 
                "shortest_path_coordinates_cl1_R200", "shortest_path_coordinates_cl2_R200", 
                "arr_mainstreet_cl1_R200", "arr_mainstreet_cl2_R200",
                "match_results_cl1","match_results_cl2",
                "random_match_results_cl1","random_match_results_cl2",
                "distances_cl1","distances_cl2",
                "pwd_flat_hist_cl1","pwd_flat_hist_cl2",
                "df_high_res_cl1","df_high_res_cl2", 
                ]

        base_path = f"saves/{exp}/precision_{cut}"
        loaded_data = {}

        for i, ff in enumerate(files):
            ## Producing pairwise distances for the entire dataset is time consuming
            ## so they are only available for high precision data for now
            if exp == 'Trace1_Location2_Cell1' and (ff == "pwd_flatten_cl1" or ff == "pwd_flatten_cl2"):
                base_path = f'saves/{exp}/precision_12.5'
            if exp == 'Trace1_Location4_Cell1' and (ff == "pwd_flatten_cl1" or ff == "pwd_flatten_cl2"):
                base_path = f'saves/{exp}/precision_10'

            file_name = f"{exp}_{ff}.pkl"
            full_path = os.path.join(base_path, file_name)
            loaded_data[ff] = viz.load_pickle(full_path)

        # Access the loaded data as variables
        #df_entire_cl1 = loaded_data["df_entire_cl1"]
        #df_entire_cl2 = loaded_data["df_entire_cl2"]
        df_entire_cl1_precise = loaded_data["df_entire_cl1_precise"]
        df_entire_cl2_precise = loaded_data["df_entire_cl2_precise"]
        #df_entire_concat = loaded_data["df_entire_concat"]
        df_cl1_com_R100 = loaded_data["df_cl1_com_R100"]
        df_cl2_com_R100 = loaded_data["df_cl2_com_R100"]
        shortest_paths_cl1_R100 = loaded_data["shortest_paths_cl1_R100"]
        shortest_paths_cl2_R100 = loaded_data["shortest_paths_cl2_R100"]
        shortest_path_coordinates_cl1_R100 = loaded_data["shortest_path_coordinates_cl1_R100"]
        shortest_path_coordinates_cl2_R100 = loaded_data["shortest_path_coordinates_cl2_R100"]
        arr_mainstreet_cl1_R100 = loaded_data["arr_mainstreet_cl1_R100"]
        arr_mainstreet_cl2_R100 = loaded_data["arr_mainstreet_cl2_R100"]
        #df_tp0_concat = loaded_data["df_tp0_concat"]
        df_cl1_com_R200 = loaded_data["df_cl1_com_R200"]
        df_cl2_com_R200 = loaded_data["df_cl2_com_R200"]
        shortest_paths_cl1_R200 = loaded_data["shortest_paths_cl1_R200"]
        shortest_paths_cl2_R200 = loaded_data["shortest_paths_cl2_R200"]
        shortest_path_coordinates_cl1_R200 = loaded_data["shortest_path_coordinates_cl1_R200"]
        shortest_path_coordinates_cl2_R200 = loaded_data["shortest_path_coordinates_cl2_R200"]
        arr_mainstreet_cl1_R200 = loaded_data["arr_mainstreet_cl1_R200"]
        arr_mainstreet_cl2_R200 = loaded_data["arr_mainstreet_cl2_R200"]
        match_results_cl1 = loaded_data["match_results_cl1"]
        match_results_cl2 = loaded_data["match_results_cl2"]
        random_match_results_cl1 = loaded_data["random_match_results_cl1"]
        random_match_results_cl2= loaded_data["random_match_results_cl2"]
        distances_cl1 = loaded_data["distances_cl1"]
        distances_cl2 = loaded_data["distances_cl2"]
        pwd_flat_hist_cl1 = loaded_data["pwd_flat_hist_cl1"]
        pwd_flat_hist_cl2 = loaded_data["pwd_flat_hist_cl2" ]
        df_high_res_cl1 = loaded_data["df_high_res_cl1"]
        df_high_res_cl2 = loaded_data["df_high_res_cl2" ]


        ## ToC
        st.sidebar.markdown("# Table of Contents")
        st.sidebar.markdown("1. [Analysis](#analysis)")
        st.sidebar.markdown("   1.1. [3D scatter plots of blinking events](#scatter-plots)")
        st.sidebar.markdown("   1.2. [3D scatter plots of blinking events grouped within a sphere of radius R=200 nm](#sphere-plots)")
        st.sidebar.markdown("   1.3. [2D plots of two different choices for the center of mass (CoM), and the shortest walk between CoM with a radius of R=200 nm](#2d-plots)")
        st.sidebar.markdown("   1.4. [Mainstreet time-point assignments for each backstreet blinking events](#mainstreet-timepoint)")
        st.sidebar.markdown("   1.5. [Predicted vs. random assigments of backstreet blinking events](#algorithm-vs-random)")
        st.sidebar.markdown("   1.6. [3D scatter plots of 20 kb resolution data after backstreet assignments](#scatter-20kb)")
        st.sidebar.markdown("2. [Appendix](#appendix)")
        st.sidebar.markdown("   2.1. [A1. Center of mass radius analysis for dimension reduction](#com-radius)")


    st.header("Analysis")
    st.markdown("<a id='analysis'></a>", unsafe_allow_html=True)
    st.subheader("1- 3D scatter plots of blinking events")
    st.markdown("<a id='scatter-plots'></a>", unsafe_allow_html=True)
    ## 3D plots 
    col1,col2 = st.columns([2,2])
    with col1:
        ## CLUSTER 1
        color_set = "Light24"
        fig11 = viz.plotly_3D(df_entire_cl1_precise,color_set,f"{exp} - {cut} - Cluster 1 data after precision cut")
        st.plotly_chart(fig11, use_container_width=True)

    with col2:
        ## CLUSTER 2
        color_set = "Light24"
        fig21 = viz.plotly_3D(df_entire_cl2_precise,color_set,f"{exp} - {cut} - Cluster 2 data after precision cut")
        st.plotly_chart(fig21, use_container_width=True)

    st.subheader("2- 3D scatter plots of blinking events grouped within a sphere of radius R=200 nm")
    st.markdown("<a id='sphere-plots'></a>", unsafe_allow_html=True)
    ## 3D plots after dim reduction 
    col1,col2 = st.columns([2,2])
    with col1:
        ## CLUSTER 1
        fig12 = viz.plotly_3D(df_cl1_com_R200,color_set,f"{exp} - {cut} - Cluster 1 canter-of-mass data")
        st.plotly_chart(fig12, use_container_width=True)
    with col2:
        ## CLUSTER 2
        fig22 = viz.plotly_3D(df_cl2_com_R200,color_set,f"{exp} - {cut} - Cluster 2 canter-of-mass data")
        st.plotly_chart(fig22, use_container_width=True)

    ## 2D plots
    st.subheader("3- 2D plots of two different choices for the center of mass (CoM), and the shortest walk between CoM with a radius of R=200 nm") 
    st.markdown("<a id='2d-plots'></a>", unsafe_allow_html=True)
    col1,col2 = st.columns([2,2])
    with col1:
        ## CLUSTER 1
        axes1 = st.multiselect('Choose a pair of axes',['x', 'y', 'z'],default=['x','y'], key='axes1')
        fig13 = viz.plot2D_subplots(df_entire_cl1_precise,df_cl1_com_R100,df_cl1_com_R200,shortest_path_coordinates_cl1_R200,"cl1",[axes1[0],axes1[1]])
        st.plotly_chart(fig13, use_container_width=True)
    with col2:
        ## CLUSTER 2
        axes2 = st.multiselect('Choose a pair of axes',['x', 'y', 'z'],default=['x','y'], key='axes2')
        fig23 = viz.plot2D_subplots(df_entire_cl2_precise,df_cl2_com_R100,df_cl2_com_R200,shortest_path_coordinates_cl2_R200,"cl2",[axes2[0],axes2[1]])
        st.plotly_chart(fig23, use_container_width=True)

    ## Backst Assignments
    st.subheader("4- Mainstreet time-point assignments for each backstreet blinking events")
    st.markdown("<a id='mainstreet-timepoint'></a>", unsafe_allow_html=True)
    col1,col2 = st.columns([2,2])
    with col1:
        ## CLUSTER 1
        fig14 = viz.plotly_backst_distibutions(match_results_cl1,df_cl1_com_R200,"cl1")
        st.plotly_chart(fig14, use_container_width=True)
        fig15 = viz.plotly_Sankey_diagram(match_results_cl1,"cl1")
        st.plotly_chart(fig15, use_container_width=True)
        norm_flag1 = st.selectbox("Normalize", [True, False], index=0, key='norm1')
        fig16 = viz.backst_dist(match_results_cl1,norm_flag1)
        st.plotly_chart(fig16, use_container_width=True)
    with col2:
        ## CLUSTER 2
        fig24 = viz.plotly_backst_distibutions(match_results_cl2,df_cl2_com_R200,"cl2")
        st.plotly_chart(fig24, use_container_width=True)
        fig25 = viz.plotly_Sankey_diagram(match_results_cl2,"cl2")
        st.plotly_chart(fig25, use_container_width=True)       
        norm_flag2 = st.selectbox("Normalize", [True, False], index=0, key='norm2')
        fig26 = viz.backst_dist(match_results_cl2,norm_flag2)
        st.plotly_chart(fig26, use_container_width=True)

    ## Backst Assignments vs Random Assignments
    st.subheader("5- Predicted vs. random assigments of backstreet blinking events")
    st.markdown("<a id='algorithm-vs-random'></a>", unsafe_allow_html=True)
    col1,col2 = st.columns([2,2])
    with col1:
        ## CLUSTER 1      
        fig17 = viz.plotly_backst_distibutions_with_randoms(match_results_cl1,df_cl1_com_R200,random_match_results_cl1,"cl1")
        st.plotly_chart(fig17, use_container_width=True)

        fig18 = viz.plotly_random_vs_prediction(distances_cl1,"cl1")
        st.plotly_chart(fig18, use_container_width=True)

        fig19 = viz.plotly_box_plot(distances_cl1,"cl1")
        st.plotly_chart(fig19, use_container_width=True)

        st.markdown(f"Mann-Whitney U test *p-value*: {viz.non_parametric_tests(distances_cl1)}")
    with col2:
        ## CLUSTER 2
        fig27 = viz.plotly_backst_distibutions_with_randoms(match_results_cl1,df_cl1_com_R200,random_match_results_cl1,"cl1")
        st.plotly_chart(fig27, use_container_width=True)

        fig28 = viz.plotly_random_vs_prediction(distances_cl2,"cl2")
        st.plotly_chart(fig28, use_container_width=True)

        fig29 = viz.plotly_box_plot(distances_cl2,"cl2")
        st.plotly_chart(fig29, use_container_width=True)
        st.markdown(f"Mann-Whitney U test *p-value*: {viz.non_parametric_tests(distances_cl2)}")

    ## 3D scatter plots of 20 kb resolution data after backstreet assignments
    st.subheader("6- 3D scatter plots of 20 kb resolution data after backstreet assignments")
    st.markdown("<a id='scatter-20kb'></a>", unsafe_allow_html=True)
    st.markdown('*Hover over data to see new time-point values \"New Time Point\".*')
    col1,col2 = st.columns([2,2])
    with col1:
        ## CLUSTER 1      
        fig110 = viz.plotly_3D_new_assignments(df_high_res_cl1,"cl1")
        st.plotly_chart(fig110, use_container_width=True)
    with col2:
        ## CLUSTER 2  
        fig210 = viz.plotly_3D_new_assignments(df_high_res_cl2,"cl2")
        st.plotly_chart(fig210, use_container_width=True)


    ## Appendix
    st.header("Appendix")
    st.subheader("A1- Center of mass radius analysis for dimension reduction")
    st.markdown('*Push \"Reset axes\" on the top right of the figure if the distribution is out of scale.*')

    ## CoM Radius analysis 
    col1,col2 = st.columns([2,2])
    with col1:
        figA11 = viz.plot_bar_histogram_data(pwd_flat_hist_cl1,"cl1")
        st.plotly_chart(figA11, use_container_width=True)
    with col2:
        figA12 = viz.plot_bar_histogram_data(pwd_flat_hist_cl2,"cl2")
        st.plotly_chart(figA12, use_container_width=True)



if __name__ == '__main__':
    main()