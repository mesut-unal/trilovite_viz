import os
import json
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.sparse import dok_matrix
from scipy.spatial import distance
from scipy.spatial.distance import cdist

import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
import plotly.colors as colors

import viz_functions as viz

import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

@st.experimental_memo

def main():
    st.set_page_config(layout="wide")

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


    dataset = st.selectbox("Dataset", ['exp1, precision cut:12.5','loc4_cell1, precision cut:0'], index=0)
    exp = dataset.split(",")[0]
    cut = dataset.split(":")[1]
    files = ["df_entire_cl1","df_entire_cl2",
            "df_entire_cl1_precise","df_entire_cl2_precise",
            "df_entire_concat",
            "df_cl1_com_R100", "df_cl2_com_R100", 
            "shortest_paths_cl1_R100", "shortest_paths_cl2_R100",
            "shortest_path_coordinates_cl1_R100", "shortest_path_coordinates_cl2_R100", 
            "arr_mainstreet_cl1_R100", "arr_mainstreet_cl2_R100", 
            "df_tp0_concat",
            "df_cl1_com_R200", "df_cl2_com_R200", 
            "shortest_paths_cl1_R200", "shortest_paths_cl2_R200", 
            "shortest_path_coordinates_cl1_R200", "shortest_path_coordinates_cl2_R200", 
            "arr_mainstreet_cl1_R200", "arr_mainstreet_cl2_R200",
            "match_results_cl1","match_results_cl2",
            "random_match_results_cl1","random_match_results_cl2",
            "distances_cl1","distances_cl2"
            ]
    base_path = "saves/"
    loaded_data = {}
    for ff in files:
        if ff == "df_entire_cl1_precise" or ff == "df_entire_cl2_precise":
            file_name = f"{exp}_{ff}_cut{cut}.pkl"
        else:
            file_name = f"{exp}_{ff}.pkl"
        full_path = os.path.join(base_path, file_name)
        loaded_data[ff] = viz.load_pickle(full_path)

    # Access the loaded data as variables
    df_entire_cl1 = loaded_data["df_entire_cl1"]
    df_entire_cl2 = loaded_data["df_entire_cl2"]
    df_entire_cl1_precise = loaded_data["df_entire_cl1_precise"]
    df_entire_cl2_precise = loaded_data["df_entire_cl2_precise"]
    df_entire_concat = loaded_data["df_entire_concat"]
    df_cl1_com_R100 = loaded_data["df_cl1_com_R100"]
    df_cl2_com_R100 = loaded_data["df_cl2_com_R100"]
    shortest_paths_cl1_R100 = loaded_data["shortest_paths_cl1_R100"]
    shortest_paths_cl2_R100 = loaded_data["shortest_paths_cl2_R100"]
    shortest_path_coordinates_cl1_R100 = loaded_data["shortest_path_coordinates_cl1_R100"]
    shortest_path_coordinates_cl2_R100 = loaded_data["shortest_path_coordinates_cl2_R100"]
    arr_mainstreet_cl1_R100 = loaded_data["arr_mainstreet_cl1_R100"]
    arr_mainstreet_cl2_R100 = loaded_data["arr_mainstreet_cl2_R100"]
    df_tp0_concat = loaded_data["df_tp0_concat"]
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



    col1,col2 = st.columns([3,2])
    with col1:
        ## CLUSTER 1
        color_set = "Light24"
        fig11 = viz.plotly_3D(df_entire_cl1_precise,color_set,"Cluster 1 data after precision cut")
        st.plotly_chart(fig11, use_container_width=True)
        fig12 = viz.plotly_3D(df_cl1_com_R200,color_set,"Cluster 1 canter-of-mass data within spheres of R=200 nm")
        st.plotly_chart(fig12, use_container_width=True)

        # 2D plots
        axes1 = st.multiselect('Choose a pair of axes',['x', 'y', 'z'],default=['x','y'], key='axes1')
        fig13 = viz.plot2D_subplots(df_entire_cl1_precise,df_cl1_com_R100,df_cl1_com_R200,shortest_path_coordinates_cl1_R200,"cl1",[axes1[0],axes1[1]])
        st.plotly_chart(fig13, use_container_width=True)

        #Backst Assignment
        fig14 = viz.plotly_backst_distibutions(match_results_cl1,df_cl1_com_R200,"cl1")
        st.plotly_chart(fig14, use_container_width=True)
        fig15 = viz.plotly_Sankey_diagram(match_results_cl1,"cl1")
        st.plotly_chart(fig15, use_container_width=True)
        fig16 = viz.backst_dist(match_results_cl1)
        st.plotly_chart(fig16, use_container_width=True)

        st.subheader("Randomly assigning backstreets for a comparison")
        fig17 = viz.plotly_backst_distibutions_with_randoms(match_results_cl1,df_cl1_com_R200,random_match_results_cl1,"cl1")
        st.plotly_chart(fig17, use_container_width=True)
        fig18 = viz.plotly_random_vs_prediction(distances_cl1,"cl1")
        st.plotly_chart(fig18, use_container_width=True)


    with col2:
        ## CLUSTER 2
        color_set = "Light24"
        fig21 = viz.plotly_3D(df_entire_cl2_precise,color_set,"Cluster 2 data after precision cut")
        st.plotly_chart(fig21, use_container_width=True)
        fig22 = viz.plotly_3D(df_cl2_com_R200,color_set,"Cluster 2 canter-of-mass data within spheres of R=200 nm")
        st.plotly_chart(fig22, use_container_width=True)

        # 2D plots
        axes2 = st.multiselect('Choose a pair of axes',['x', 'y', 'z'],default=['x','y'], key='axes2')
        fig23 = viz.plot2D_subplots(df_entire_cl2_precise,df_cl2_com_R100,df_cl2_com_R200,shortest_path_coordinates_cl2_R200,"cl2",[axes2[0],axes2[1]])
        st.plotly_chart(fig23, use_container_width=True)

        #Backst Assignment
        fig24 = viz.plotly_backst_distibutions(match_results_cl2,df_cl2_com_R200,"cl2")
        st.plotly_chart(fig24, use_container_width=True)
        fig25 = viz.plotly_Sankey_diagram(match_results_cl2,"cl2")
        st.plotly_chart(fig25, use_container_width=True)
        fig26 = viz.backst_dist(match_results_cl2)
        st.plotly_chart(fig26, use_container_width=True)

        st.subheader("Randomly assigning backstreets for a comparison")
        fig27 = viz.plotly_backst_distibutions_with_randoms(match_results_cl1,df_cl1_com_R200,random_match_results_cl1,"cl1")
        st.plotly_chart(fig27, use_container_width=True)
        fig28 = viz.plotly_random_vs_prediction(distances_cl2,"cl2")
        st.plotly_chart(fig28, use_container_width=True)

if __name__ == '__main__':
    main()