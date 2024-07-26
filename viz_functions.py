
import os
import json
import math
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
# from scipy.sparse import dok_matrix
# from scipy.spatial import distance
# from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu#,kruskal, chi2_contingency
from scipy.spatial import KDTree

import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
import plotly.colors as colors
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
# from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from googleapiclient.discovery import build
from google.oauth2 import service_account
import googleapiclient.discovery
from base64 import b64decode
from datetime import date, datetime

def load_pickle(input):
    pfile = open(input, 'rb')
    ffile = pd.read_pickle(pfile)
    pfile.close()
    return ffile

def non_parametric_tests(dist,dist_random):
    # # Kolmogorov-Smirnov test
    # ks_stat, ks_p_value = ks_2samp(distances['pred_dist'], distances['random_dist'])
    # # print(f"KS Statistic: {ks_stat}, p-value: {ks_p_value}")

    # Mann-Whitney U test
    mwu_stat, mwu_p_value = mannwhitneyu(dist['distance'], dist_random['distance']) 
    # print(f"Mann-Whitney U Statistic: {mwu_stat}, p-value: {mwu_p_value}")

    # # Kruskal-Wallis test
    # kw_stat, kw_p_value = kruskal(distances['pred_dist'], distances['random_dist'])
    # print(f"Kruskal-Wallis Statistic: {kw_stat}, p-value: {kw_p_value}")

    # # Chi-squared test (for categorical data)
    # # Example assuming you have a DataFrame with categorical variables, modify as needed
    # observed = pd.crosstab(distances['pred_dist'], distances['random_dist'])
    # chi2_stat, chi2_p_value, _, _ = chi2_contingency(observed)
    # print(f"Chi-squared Statistic: {chi2_stat}, p-value: {chi2_p_value}")

    return mwu_p_value

###############
## VISULAZATION 
###############
def display_image_with_placeholder(image_path, caption, placeholder_path='trilobite-fossils.jpg'):
    if os.path.exists(image_path):
        st.image(image_path, caption=caption)
    else:
        st.image(placeholder_path, caption='Image not available')

def plotly_3D(df_subset,color_set,title):
    # Define the time-point classes
    time_point_classes = sorted(df_subset['time-point'].unique())

    # Create a color palette with 20 distinct colors
    palette = getattr(colors.qualitative, color_set)

    # Assign colors from the palette to each time-point class
    color_map = dict(zip(time_point_classes, palette))

    # Create a scatter plot for each entry with hover labels and colored by time-point class
    fig = go.Figure()

    for time_point_class, color in color_map.items():
        subset = df_subset[df_subset['time-point'] == time_point_class]
        fig.add_trace(go.Scatter3d(
            x=subset['x'],
            y=subset['y'],
            z=subset['z'],
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                opacity=0.8
            ),
            text=subset.apply(lambda row: f"Image-ID: {row['image-ID']}, Time-Point: {row['time-point']}, X: {row['x']}, Y: {row['y']}, Z: {row['z']}", axis=1),
            hovertemplate='<b>%{text}</b>',
            name=f"Time-Point {time_point_class}",
            visible=True if time_point_class == time_point_classes[0] else 'legendonly'  # Set visibility to 'legendonly' for all except the first
        ))

    # Define the click event handler
    def on_click(trace, points, state):
        if points.point_inds:
            selected_points = points.point_inds
            selected_data = df_subset.iloc[selected_points]
            # Do something with the selected data, such as displaying it in a table or updating other visualizations

    # Assign the event handler to the scatter plot
    fig.data[0].on_click(on_click)

    # Set plot layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        title=f'{title}',
        width=900,  # Adjust the width of the canvas as desired
        height=900,  # Adjust the height of the canvas as desired
        showlegend=True,  # Show the legend
        legend=dict(title='Time-Point')  # Set the legend title
    )

    # Show the plot
    return fig

def plot_3d_time_series_with_dropdown(df_entire, match_result):
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Define the color map
    color_map = ['#ffffe0', '#ffbcaf', '#f4777f', '#cf3759', '#93003a']

    time_points = sorted(df_entire['time-point'].unique())
    traces_per_time_point = []

    # Prepare the traces
    for time_point in time_points:
        df_filtered = df_entire[df_entire['time-point'] == time_point]
        match_filtered = match_result[match_result['matching_point_time_point'] == time_point]
        backstreet_points = match_filtered['backstreet_time_point'].unique()
        
        # df_entire trace
        fig.add_trace(
            go.Scatter3d(
                x=df_filtered['x'],
                y=df_filtered['y'],
                z=df_filtered['z'],
                mode='markers',
                marker=dict(size=3, color='#00429d'),
                name=f'MS - Time-point {time_point}',
                legendgroup=f'entire_{time_point}',
                visible=False
            )
        )
        # Count how many traces are added for each time point (1 for entire + each backstreet time point)
        num_traces_this_time_point = 1  # start with df_entire

        # match_result traces
        for idx, backstreet_time_point in enumerate(backstreet_points):
            subgroup = match_filtered[match_filtered['backstreet_time_point'] == backstreet_time_point]
            fig.add_trace(
                go.Scatter3d(
                    x=subgroup['x'],
                    y=subgroup['y'],
                    z=subgroup['z'],
                    mode='markers',
                    marker=dict(size=5, color=color_map[idx % len(color_map)]),
                    name=f'BS - Time-point {time_point} (Backstreet {backstreet_time_point})',
                    legendgroup=f'match_{time_point}_{backstreet_time_point}',
                    visible=False
                )
            )
            num_traces_this_time_point += 1

        traces_per_time_point.append(num_traces_this_time_point)

    # Create the dropdown menus
    total_traces = sum(traces_per_time_point)
    dropdown_menus = []
    current_index = 0

    for i, time_point in enumerate(time_points):
        visible = [False] * total_traces
        for j in range(traces_per_time_point[i]):
            visible[current_index + j] = True
        dropdown_menus.append(
            dict(
                args=["visible", visible],
                label=f"Time-point {time_point}",
                method="restyle"
            )
        )
        current_index += traces_per_time_point[i]

    # Update layout with dropdown
    fig.update_layout(
        title='3D Time Series Plot with Time-point Selection',
        scene=dict(
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            zaxis=dict(title='z')
        ),
        width=1000, height=1000,
        updatemenus=[{
            "buttons": dropdown_menus,
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top"
               }]
        )

    # Set initial visibility
    initial_visibility = [False] * total_traces
    if traces_per_time_point:
        initial_visibility[:traces_per_time_point[0]] = [True] * traces_per_time_point[0]
    for trace in fig.data[:traces_per_time_point[0]]:
        trace.visible = True  # Set the first time-point traces to be visible initially

    return fig


def plot2D_subplots(df_all,df_com,shortest_path_coordinates,tr_info,axis,MAINSTREET_TP_RANGE):

    # Create a DataFrame from data_points
    df_com_filter = df_com[["image-ID","time-point","x","y","z","precisionx","precisiony","precisionz"]][df_com['time-point'] < MAINSTREET_TP_RANGE[1]+1]
    # Initialize an empty DataFrame to store the results
    df_shortest_paths = pd.DataFrame(columns=['x', 'y', 'z', 'time-point'])
    i = 0
    for time_point, data in df_all[df_all['time-point'] < MAINSTREET_TP_RANGE[1]+1].groupby('time-point'):
        # Create a DataFrame for the shortest path
        df_shortest_path = pd.DataFrame(shortest_path_coordinates[i], columns=['x', 'y', 'z'])
        df_shortest_path['time-point'] = time_point  # Assign 'time-point' to all rows
        df_shortest_paths = pd.concat([df_shortest_paths, df_shortest_path], ignore_index=True)
        i += 1

    # Get the number of time-points and divide by 2 to determine the number of rows
    n_time_points = len(df_all[df_all['time-point'] < MAINSTREET_TP_RANGE[1]+1]['time-point'].unique())
    n_rows = (n_time_points + 1) // 2  # Add 1 and floor division to get even number of rows

    # Create a subplot with a 2xN layout
    row_width = [1.0] * n_rows
    column_width = [0.5] * 2
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=[f'Time-Point {i}' for i,el in df_all[df_all['time-point'] < MAINSTREET_TP_RANGE[1]+1].groupby('time-point')],
        row_width=row_width,
        column_width=column_width
    )

    row, col = 1, 1  # Start with the first subplot
    for time_point, data in df_all[df_all['time-point'] < MAINSTREET_TP_RANGE[1]+1].groupby('time-point'):
        # Create your scatter plots and add them to the subplot grid
        trace_scatter = go.Scatter(
            x=df_com_filter[df_com_filter['time-point'] == time_point][axis[0]],
            y=df_com_filter[df_com_filter['time-point'] == time_point][axis[1]],
            mode='markers',
            marker=dict(color='darkorange', size=8, opacity=1.0),
            marker_symbol='circle-open',
            name=f'CoM Time-Point {time_point} for R=100'
        )

        trace_shortest_path = go.Scatter(
            x=df_shortest_paths[df_shortest_paths['time-point'] == time_point][axis[0]],
            y=df_shortest_paths[df_shortest_paths['time-point'] == time_point][axis[1]],
            mode='lines',
            line=dict(color='red', width=2),
            opacity=0.5,
            name=f'Shortest Path TP={time_point}'
        )

        trace_contour_data = go.Histogram2d(
            x=df_all[df_all['time-point'] == time_point][axis[0]],
            y=df_all[df_all['time-point'] == time_point][axis[1]],
            colorscale='Greys',  # Color scale
            showscale=False,  # Show color scale
        )

        fig.add_trace(trace_contour_data, row=row, col=col)
        fig.add_trace(trace_scatter, row=row, col=col)
        fig.add_trace(trace_shortest_path, row=row, col=col)
        
        fig.update_xaxes(title_text=f'{axis[0]} axis', row=row, col=col)
        fig.update_yaxes(title_text=f'{axis[1]} axis', row=row, col=col)

        col += 1
        if col > 2:
            col = 1
            row += 1

    fig.update_layout(
    title_text=f"{tr_info[-1]}",
    title_x=0.1,  # Center the title horizontally
    title_y=0.99,  # Adjust the vertical position of the title
    title_font=dict(size=24),  # Set the font size
    )
    # Show the subplots
    fig.update_layout(height=2500, width=1000)
    
    return fig

def plotly_backst_distibutions(match_results,df_com,tr_info,MAINSTREET_TP_RANGE):
    fig = go.Figure()


    fig.add_trace(go.Histogram(x=df_com[df_com['time-point']<MAINSTREET_TP_RANGE[1]+1]["time-point"], 
                                #opacity=0.5, 
                        name = "Mainstreet",
                        nbinsx=int((MAINSTREET_TP_RANGE[1]+1-0)/1)
                        )
                )

    fig.add_trace(go.Histogram(x=match_results["matching_point_time_point"], 
                            name='Backstreet',
                            nbinsx=int((MAINSTREET_TP_RANGE[1]+1-0)/1)
                    )
                )




    fig.update_layout(barmode='overlay',
                    template = "ggplot2",
                    width=1000, height=400,
                    title=f'{tr_info[-1]} - Backstreet predictions',
                    title_x=0.1,  
                    title_y=0.9, 
                    title_font=dict(size=20), 
                    )

    fig.update_yaxes(type="log")

    fig.update_xaxes(title_text="Time points")  # Change the x-axis title
    fig.update_yaxes(title_text="Number of entries")  # Change the y-axis title

    return fig


def plotly_Sankey_diagram(match_results, tr_info):
    # Assign unique colors to backstreet_time_point and matching_point_time_point
    colors = px.colors.qualitative.D3 + px.colors.qualitative.Light24
    color_map = dict(zip(match_results['backstreet_time_point'].unique(), colors[:len(match_results['backstreet_time_point'].unique())]))
    match_results['source_node_color'] = match_results['backstreet_time_point'].map(color_map)
    color_map = dict(zip(match_results['matching_point_time_point'].unique(), colors[:len(match_results['matching_point_time_point'].unique())]))
    match_results['target_node_color'] = match_results['matching_point_time_point'].map(color_map)



    result = match_results.groupby(['backstreet_time_point', 'matching_point_time_point', 'source_node_color', 'target_node_color']).size().reset_index(name='num_lines')

    # Now, 'result' DataFrame contains the number of lines between each source and target

    # Create a list of unique source and target nodes
    nodes = pd.concat([result['backstreet_time_point'], result['matching_point_time_point']]).unique()

    # Create a mapping from nodes to indices
    node_indices = {node: index for index, node in enumerate(nodes)}

    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color = pd.concat([result['source_node_color'], result['target_node_color']], ignore_index=True)
        ),
        link=dict(
            source=result['backstreet_time_point'].map(node_indices),
            target=result['matching_point_time_point'].map(node_indices),
            value=result['num_lines'],
            color=result['source_node_color']
        )
    ))

    fig.update_layout(
        title=f'{tr_info[-1]} - flow of backstreet assignments',
        xaxis_title="Source",
        yaxis_title="Target",
    )

    return fig

def backst_dist(match_results,norm_flag):
    grouped = match_results.groupby('matching_point_time_point')['backstreet_time_point'].value_counts(normalize=norm_flag).unstack()

    # fig, ax = plt.subplots()
    # ax = grouped.plot(kind='bar', stacked=True)
    
    # plt.xlabel('matching_point_time_point')
    # plt.ylabel('Count')
    # plt.title('Percentage of backst_timbackstreet_time_pointe_point values for each matching_point_time_point')
    # plt.legend(title='backstreet_time_point', bbox_to_anchor=(1.05, 1), loc='upper left')

    # st.pyplot(fig)

    data = []
    for col in grouped.columns:
        trace = go.Bar(
            x=grouped.index,
            y=grouped[col],
            name=str(col)
        )
        data.append(trace)

    # Layout for the Plotly chart
    layout = go.Layout(
        barmode='stack',
        xaxis=dict(title='matching_point_time_point'),
        yaxis=dict(title='Count'),
        title='Ratios of backstreet data values for each matching mainstreet time_point',
        legend=dict(title='backstreet_time_point', x=1.05, y=1, traceorder='normal', orientation='h')
    )

    # Create the Plotly figure
    fig = go.Figure(data=data, layout=layout)
    return fig

def plotly_backst_distibutions_with_randoms(match_results,df_com,random_match_results,tr_info,MAINSTREET_TP_RANGE):
  trace_hist1 = go.Histogram(x=df_com[df_com['time-point']<MAINSTREET_TP_RANGE[1]+1]["time-point"], 
                             opacity=0.7, 
                             marker_color='green', 
                             name='Main',
                             nbinsx=int((MAINSTREET_TP_RANGE[1]+1-MAINSTREET_TP_RANGE[0])/1)
                             )
  trace_hist2 = go.Histogram(x=match_results["matching_point_time_point"], 
                             opacity=0.5, 
                             marker=dict(color='blue', line=dict(color='blue', width=2)), 
                             name='Prediction',
                             nbinsx=int((MAINSTREET_TP_RANGE[1]+1-MAINSTREET_TP_RANGE[0])/1)
                             )
  trace_hist3 = go.Histogram(
                            x=random_match_results["matching_point_time_point"],
                            histfunc='count',
                            opacity=1,
                            marker=dict(color='red', line=dict(color='red', width=2)),
                            name='Random Assignment',
                            nbinsx=int((MAINSTREET_TP_RANGE[1]+1-MAINSTREET_TP_RANGE[0])/1)
                            )

  layout = go.Layout(barmode='overlay')

  fig = go.Figure(data=[trace_hist3,trace_hist2, trace_hist1 ], layout=layout)

  fig.update_layout(
                  template = "ggplot2",
                  width=1000, height=400,
                  title=f"{tr_info[-1]}",
                  title_x=0.1,  
                  title_y=0.9, 
                  title_font=dict(size=20), 
                    )

  fig.update_yaxes(type="log")

  fig.update_xaxes(title_text="Time points")  # Change the x-axis title
  fig.update_yaxes(title_text="Number of entries")  # Change the y-axis title

  return fig

def plotly_random_vs_prediction(distances,distances2,tr_info,MAINSTREET_TP_RANGE):
    trace_hist1 = go.Histogram(x=distances['distance'], 
                                opacity=0.7, 
                                marker_color='green', 
                                name='Prediction',
                                nbinsx=int((MAINSTREET_TP_RANGE[1]+1-MAINSTREET_TP_RANGE[0])/1)
                                )
    trace_hist2 = go.Histogram(x=distances2['distance'], 
                                opacity=0.5, 
                                marker=dict(color='darkorange', line=dict(color='darkorange', width=2)), 
                                name='Random Assignment',
                                nbinsx=int((MAINSTREET_TP_RANGE[1]+1-MAINSTREET_TP_RANGE[0])/1)
                                )

    traces = [trace_hist1,trace_hist2]

    # Optionally add a third histogram 
    # if not distances.empty:
    #     trace_hist3 = go.Histogram(x=distances['distance'], 
    #                                 opacity=0.6, 
    #                                 marker_color='blue', 
    #                                 name='Prediction R=100 nm',
    #                                 nbinsx=int((MAINSTREET_TP_RANGE[1]+1-MAINSTREET_TP_RANGE[0])/1)
    #                                 )
    #     traces.append(trace_hist3)

    layout = go.Layout(barmode='overlay')
    fig = go.Figure(data=traces, layout=layout)

    fig.update_layout(
                  template = "ggplot2",
                  width=1000, height=400,
                  title=f"{tr_info[-1]}",
                  title_x=0.1,  
                  title_y=0.9, 
                  title_font=dict(size=20), 
                    )

    fig.update_xaxes(title_text="Distance (nm)")  # Change the x-axis title
    fig.update_yaxes(title_text="Number of entries")  # Change the y-axis title

    return fig

def plotly_box_plot(dist,dist_random,tr_info):
    # Melt the prediction distances with uniform column name 'distance'
    melted_distances = dist.melt(value_vars=['distance'],
                                      var_name='Assignment Type', value_name='Distance')
    melted_distances['Assignment Type'] = 'Prediction'

    # Melt the random distances with uniform column name 'distance'
    melted_distances_random = dist_random.melt(value_vars=['distance'],
                                                    var_name='Assignment Type', value_name='Distance')
    melted_distances_random['Assignment Type'] = 'Random'

    # Concatenate the melted DataFrames
    melted_distances = pd.concat([melted_distances, melted_distances_random], ignore_index=True)

    fig = px.box(melted_distances, y='Distance', color='Assignment Type',
                labels={'Assignment Type': 'Assignment Type', 'Distance': 'Distance [nm]'},
                title=f'Box plot for predicted and random assignments for Trace {tr_info[-1]}')

    return fig

def pwd_histograms(hist_data,tr_info):
    ## NOT CURRENTLY IN USE
    ## SUCCEDED BY plot_bar_histogram_data DUE TO FILE SIZE ISSUES
    # Create animated histogram using Plotly Express
    histogram = px.histogram(hist_data, x='Pairwise Distance', animation_frame='time-point', #marginal='violin',
                    nbins=math.ceil(math.sqrt(len(hist_data)/20)), range_x=[0, hist_data['Pairwise Distance'].max()])

    histogram.update_xaxes(
        range=[0, 1200]
    )

    histogram.update_layout(
        xaxis_title='Pairwise Distance [nm]',
        yaxis_title='Count',
        title=f'{tr_info[-1]} - Pairwise Distances for Each Time-Point',
        showlegend=False
    )

    return histogram

def plot_bar_histogram_data(hist_data_saved,tr_info):
    ## Plot pairwise distances histograms from saved histogram data directly
    ## saved histogram data occupies significantly less space than dataframe objects
    histogram = px.bar(
        hist_data_saved,
        x='bin_edges',
        y='bin_values',
        labels={'bin_edges': 'Pairwise Distance [nm]', 'bin_values': 'Count'},
        animation_frame='time-point',
        range_x=[0, hist_data_saved['bin_edges'].max()],
        category_orders={'time-point': sorted(hist_data_saved['time-point'].unique())},
    )

    histogram.update_xaxes(
        range=[0, 1200]
    )

    histogram.update_layout(
        xaxis_title='Pairwise Distance [nm]',
        yaxis_title='Count',
        title=f'Pairwise Distances for Each Time-Point for {tr_info[-1]}',
        showlegend=False,
        template = "ggplot2"
    )

    return histogram

def plotly_3D_new_assignments(df_high_res,tr_info):
    # order = df_high_res.sort_values(by=['predicted-time-point', 'old-time-point'])['old-time-point'].unique()
    # df_high_res['old-time-point'] = pd.Categorical(df_high_res['old-time-point'], categories=order, ordered=True)
    
    # # Map each unique 'old-time-point' to a color
    # color_map = dict(zip(df_high_res['old-time-point'].unique(), colors.qualitative.Plotly))
    # df_high_res['point_color'] = df_high_res['old-time-point'].map(color_map)

    # # Create a custom color scale based on the sorted order of 'predicted-time-point'
    # custom_color_scale = [color_map[time_point] for time_point in order]

    #df_high_res = df_high_res.sort_values(by=['predicted-time-point', 'old-time-point'])

    df_high_res_copy = df_high_res.copy()

    df_high_res_copy["predicted-time-point"] = df_high_res_copy["predicted-time-point"].astype('category')
    df_high_res_copy["old-time-point"] = df_high_res_copy["old-time-point"].astype('category')
    df_high_res_copy["new-time-point"] = df_high_res_copy["new-time-point"].astype('category')

    # Map each unique 'old-time-point' to a color
    unique_old_time_points = sorted(df_high_res_copy['old-time-point'].unique())
    color_map = dict(zip(unique_old_time_points, colors.qualitative.Plotly))
    df_high_res_copy['point_color'] = df_high_res_copy['old-time-point'].map(color_map)


    fig = px.scatter_3d(df_high_res_copy, x='x', y='y', z='z', color='old-time-point', 
                        animation_frame='predicted-time-point',
                        color_discrete_map=color_map,
                        # color_discrete_sequence=custom_color_scale,  # Use custom color scale
                        title='3D Scatter Plot with Animation Frames',
                        labels={'old-time-point': 'Old Time Point', 'predicted-time-point': 'Predicted Time Point',
                                'new-time-point': 'New Time Point', 'x': 'X', 'y': 'Y', 'z': 'Z'},
                        hover_data=['old-time-point', 'predicted-time-point', 'new-time-point', 'x', 'y', 'z'],
                        )

    fig.update_layout(scene=dict(aspectmode='data'))
    fig.update_traces(marker_size = 4)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        title=f'3D Scatter Plot for Cluster {tr_info[-1]}',
        width=900,  
        height=900, 
        showlegend=True, 
        legend=dict(title='Time-Point') 
    )         
    return fig

def plotly_pwd_histogram_with_dropdown(histograms):
    fig = go.Figure()
    buttons = []
    for time_point, (bins, histogram) in histograms.items():
        fig.add_trace(go.Bar(
            x=bins[:-1],
            y=histogram,
            width=bins[1] - bins[0],
            marker_color='#330C73',
            opacity=0.75,
            name=f'time-point {time_point}',
            visible=False  # All traces initially hidden
        ))
        buttons.append(dict(
            label=f'time-point {time_point}',
            method='update',
            args=[{'visible': [tp == time_point for tp in histograms.keys()]},
                {'title': f'Histogram of pairwise distances for time-point {time_point}'}]
        ))
    # Set the first time-point trace as visible
    fig.data[0].visible = True
    # Add dropdown menu to the figure
    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True
        }]
    )
    # Update layout
    fig.update_layout(
        title='Histogram of pairwise distances',
        xaxis_title='Pairwise distance (nm)',
        yaxis_title='Count',
        bargap=0.2,
        bargroupgap=0.1
    )
    return fig

def compute_histogram_per_tp(df, bin_size=2, max_distance = 150):
    histograms = {}
    time_points = df['time-point'].unique()
    for time_point in time_points:
        df_filtered = df[df['time-point'] == time_point]
        coords = df_filtered[['x', 'y', 'z']].values
        kdtree = KDTree(coords)
        
        # Initialize histogram bins
        bins = np.arange(0, max_distance + bin_size, bin_size)
        histogram = np.zeros(len(bins) - 1)
        
        total_points = len(coords)
        for i in range(total_points):
            distances, _ = kdtree.query(coords[i], k=total_points, distance_upper_bound=max_distance)
            # distances = distances[1:]  # Skip distance to itself (0)
            distances = distances[distances <= max_distance]  # Filter out distances greater than max_distance
            # Update histogram
            hist, _ = np.histogram(distances, bins=bins)
            histogram += hist
            
            # Print event counter every 1000 rows
            # if (i + 1) % 1000 == 0:
            #     print(f'Processed {i + 1} out of {total_points} rows for time-point {time_point}')
        histograms[time_point] = (bins, histogram)    
    return histograms



####
### Functions to append data to the Google Sheet

def get_google_credentials():
    # Accessing the base64-encoded Google credentials from a TOML section
    creds_base64 = st.secrets["google_credentials"]["credentials_base64"]
    creds_json = b64decode(creds_base64).decode("utf-8")
    creds_dict = json.loads(creds_json)
    return service_account.Credentials.from_service_account_info(creds_dict)

def append_data_to_sheet(data, spreadsheet_id, range_name):

    ## LOCAL
    # # Path to your service account key file
    # SERVICE_ACCOUNT_FILE = '/Users/munal/Documents/srm-streamlit-feedbackform-955a0d41ed0f.json'

    # # Define the scopes
    # SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    # # Load the service account credentials
    # creds = service_account.Credentials.from_service_account_file(
    #     SERVICE_ACCOUNT_FILE,
    #     scopes=SCOPES)
    ## LOCAL

    ## STREAMLIT CLOUD
    creds = get_google_credentials()
    ## STREAMLIT CLOUD

    # Convert date and datetime objects in data to string in ISO format
    formatted_data = [[item.isoformat() if isinstance(item, (date, datetime)) else item for item in row] for row in data]

    # Build the service
    service = googleapiclient.discovery.build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    # Prepare the request and append the formatted data
    request = sheet.values().append(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption="USER_ENTERED",
        body={"values": formatted_data}
    ).execute()