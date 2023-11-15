
import json
import pandas as pd
import numpy as np
import networkx as nx
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
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

def load_pickle(input):
    pfile = open(input, 'rb')
    ffile = pickle.load(pfile)
    pfile.close()
    return ffile

###############
## VISULAZATION 
###############

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
        visible='legendonly'  # Set visibility to 'legendonly' by default
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


def plot2D_subplots(df_all,df_com,df_com2,shortest_path_coordinates,cl_info,axis):

    # Create a DataFrame from data_points
    df_com_filter_R100 = df_com[["image-ID","time-point","x","y","z","precisionx","precisiony","precisionz"]][df_com['time-point'] < 20]
    df_com_filter_R200 = df_com2[["image-ID","time-point","x","y","z","precisionx","precisiony","precisionz"]][df_com2['time-point'] < 20]
    # Initialize an empty DataFrame to store the results
    df_shortest_paths = pd.DataFrame(columns=['x', 'y', 'z', 'time-point'])
    i = 0
    for time_point, data in df_all[df_all['time-point'] < 20].groupby('time-point'):
        # Create a DataFrame for the shortest path
        df_shortest_path = pd.DataFrame(shortest_path_coordinates[i], columns=['x', 'y', 'z'])
        df_shortest_path['time-point'] = time_point  # Assign 'time-point' to all rows
        df_shortest_paths = df_shortest_paths.append(df_shortest_path, ignore_index=True)
        i += 1

    # Get the number of time-points and divide by 2 to determine the number of rows
    n_time_points = len(df_all[df_all['time-point'] < 20]['time-point'].unique())
    n_rows = (n_time_points + 1) // 2  # Add 1 and floor division to get even number of rows

    # Create a subplot with a 2xN layout
    row_width = [1.0] * n_rows
    column_width = [0.5] * 2
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=[f'Time-Point {i}' for i,el in df_all[df_all['time-point'] < 20].groupby('time-point')],
        row_width=row_width,
        column_width=column_width
    )

    row, col = 1, 1  # Start with the first subplot
    for time_point, data in df_all[df_all['time-point'] < 20].groupby('time-point'):
        # Create your scatter plots and add them to the subplot grid
        trace_scatter = go.Scatter(
            x=df_com_filter_R100[df_com_filter_R100['time-point'] == time_point][axis[0]],
            y=df_com_filter_R100[df_com_filter_R100['time-point'] == time_point][axis[1]],
            mode='markers',
            marker=dict(color='darkorange', size=8, opacity=1.0),
            marker_symbol='circle-open',
            name=f'CoM Time-Point {time_point} for R=100'
        )

        trace_scatter2 = go.Scatter(
            x=df_com_filter_R200[df_com_filter_R200['time-point'] == time_point][axis[0]],
            y=df_com_filter_R200[df_com_filter_R200['time-point'] == time_point][axis[1]],
            mode='markers',
            marker=dict(color='green', size=8, opacity=1.0),
            name=f'CoM Time-Point {time_point} for R=200'
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
        fig.add_trace(trace_scatter2, row=row, col=col)
        fig.add_trace(trace_shortest_path, row=row, col=col)
        


        fig.update_xaxes(title_text=f'{axis[0]} axis', row=row, col=col)
        fig.update_yaxes(title_text=f'{axis[1]} axis', row=row, col=col)

        col += 1
        if col > 2:
            col = 1
            row += 1

    canvas_title = 'Cluster 1' if cl_info == 'cl1' else ('Cluster 2' if cl_info == 'cl2' else 'Default Title')

    fig.update_layout(
    title_text=canvas_title,
    title_x=0.1,  # Center the title horizontally
    title_y=0.99,  # Adjust the vertical position of the title
    title_font=dict(size=24),  # Set the font size
    )
    # Show the subplots
    fig.update_layout(height=2500, width=1000)
    
    return fig

def plotly_backst_distibutions(match_results,df_com,cl_info):
  fig = go.Figure()

  fig.add_trace(go.Histogram(x=match_results["matching_line_time_point"], 
                            name='Backstreet',
                            nbinsx=int((20-0)/1)
                    )
              )

  fig.add_trace(go.Histogram(x=df_com[df_com['time-point']<20]["time-point"], 
                      name = "Mainstreet R=200",
                      nbinsx=int((20-0)/1)
                    )
              )

  if cl_info == 'cl1':
      canvas_title = 'Cluster 1'
  elif cl_info == 'cl2':
      canvas_title = 'Cluster 2'
  

  fig.update_layout(barmode='overlay',
                  template = "ggplot2",
                  width=1000, height=400,
                  title=f'{canvas_title} - backstreet predictions',
                  title_x=0.1,  
                  title_y=0.9, 
                  title_font=dict(size=20), 
                    )

  fig.update_yaxes(type="log")

  fig.update_xaxes(title_text="Time points")  # Change the x-axis title
  fig.update_yaxes(title_text="Number of entries")  # Change the y-axis title

  return fig


def plotly_Sankey_diagram(match_results, cl_info):
    # Assign unique colors to backst_time_point and matching_line_time_point
    colors = px.colors.qualitative.D3 + px.colors.qualitative.Light24
    color_map = dict(zip(match_results['backst_time_point'].unique(), colors[:len(match_results['backst_time_point'].unique())]))
    match_results['source_node_color'] = match_results['backst_time_point'].map(color_map)
    color_map = dict(zip(match_results['matching_line_time_point'].unique(), colors[:len(match_results['matching_line_time_point'].unique())]))
    match_results['target_node_color'] = match_results['matching_line_time_point'].map(color_map)



    result = match_results.groupby(['backst_time_point', 'matching_line_time_point', 'source_node_color', 'target_node_color']).size().reset_index(name='num_lines')

    # Now, 'result' DataFrame contains the number of lines between each source and target

    # Create a list of unique source and target nodes
    nodes = pd.concat([result['backst_time_point'], result['matching_line_time_point']]).unique()

    # Create a mapping from nodes to indices
    node_indices = {node: index for index, node in enumerate(nodes)}

    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=result['source_node_color'].append(result['target_node_color'])
        ),
        link=dict(
            source=result['backst_time_point'].map(node_indices),
            target=result['matching_line_time_point'].map(node_indices),
            value=result['num_lines'],
            color=result['source_node_color']
        )
    ))

    if cl_info == 'cl1':
      canvas_title = 'Cluster 1'
    elif cl_info == 'cl2':
      canvas_title = 'Cluster 2'

    fig.update_layout(
        title=f'{canvas_title} - flow of backstreet assignments',
        xaxis_title="Source",
        yaxis_title="Target",
    )

    return fig

def backst_dist(match_results):
    grouped = match_results.groupby('matching_line_time_point')['backst_time_point'].value_counts(normalize=False).unstack()

    # fig, ax = plt.subplots()
    # ax = grouped.plot(kind='bar', stacked=True)
    
    # plt.xlabel('matching_line_time_point')
    # plt.ylabel('Count')
    # plt.title('Percentage of backst_time_point values for each matching_line_time_point')
    # plt.legend(title='backst_time_point', bbox_to_anchor=(1.05, 1), loc='upper left')

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
        xaxis=dict(title='matching_line_time_point'),
        yaxis=dict(title='Count'),
        title='Ratios of backstreet data values for each matching mainstreet time_point',
        legend=dict(title='backst_time_point', x=1.05, y=1, traceorder='normal', orientation='h')
    )

    # Create the Plotly figure
    fig = go.Figure(data=data, layout=layout)
    return fig

def plotly_backst_distibutions_with_randoms(match_results,df_com,random_match_results,cl_info):
  trace_hist1 = go.Histogram(x=df_com[df_com['time-point']<20]["time-point"], 
                             opacity=0.7, 
                             marker_color='green', 
                             name='Main',
                             nbinsx=int((20-0)/1)
                             )
  trace_hist2 = go.Histogram(x=match_results["matching_line_time_point"], 
                             opacity=0.5, 
                             marker=dict(color='blue', line=dict(color='blue', width=2)), 
                             name='Prediction',
                             nbinsx=int((20-0)/1)
                             )
  trace_hist3 = go.Histogram(
                            x=random_match_results["matching_line_time_point"],
                            histfunc='count',
                            opacity=1,
                            marker=dict(color='red', line=dict(color='red', width=2)),
                            name='Random Assignment',
                            nbinsx=int((20 - 0) / 1)
                            )

  layout = go.Layout(barmode='overlay')

  fig = go.Figure(data=[trace_hist3,trace_hist2, trace_hist1 ], layout=layout)

  if cl_info == 'cl1':
      canvas_title = 'Cluster 1'
  elif cl_info == 'cl2':
      canvas_title = 'Cluster 2'

  fig.update_layout(
                  template = "ggplot2",
                  width=1000, height=400,
                  title=canvas_title,
                  title_x=0.1,  
                  title_y=0.9, 
                  title_font=dict(size=20), 
                    )

  fig.update_yaxes(type="log")

  fig.update_xaxes(title_text="Time points")  # Change the x-axis title
  fig.update_yaxes(title_text="Number of entries")  # Change the y-axis title

  return fig

def plotly_random_vs_prediction(distances,cl_info):
  trace_hist1 = go.Histogram(x=distances['pred_dist'], 
                             opacity=0.7, 
                             marker_color='green', 
                             name='Prediction',
                             nbinsx=int((20-0)/1)
                             )
  trace_hist2 = go.Histogram(x=distances['random_dist'], 
                             opacity=0.5, 
                             marker=dict(color='darkorange', line=dict(color='darkorange', width=2)), 
                             name='Random Assignment',
                             nbinsx=int((20-0)/1)
                             )

  layout = go.Layout(barmode='overlay')

  fig = go.Figure(data=[trace_hist1, trace_hist2], layout=layout)

  if cl_info == 'cl1':
      canvas_title = 'Cluster 1'
  elif cl_info == 'cl2':
      canvas_title = 'Cluster 2'

  fig.update_layout(
                  template = "ggplot2",
                  width=1000, height=400,
                  title=canvas_title,
                  title_x=0.1,  
                  title_y=0.9, 
                  title_font=dict(size=20), 
                    )

  fig.update_xaxes(title_text="Distance [nm]")  # Change the x-axis title
  fig.update_yaxes(title_text="Number of entries")  # Change the y-axis title

  return fig