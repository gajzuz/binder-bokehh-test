# Libarary imports
# System paths
from os.path import join, dirname

# Data wrangling
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# Network graph 
from itertools import combinations_with_replacement    # 
import networkx as nx                                  # graphing
from networkx.algorithms import community              # address community via nx.algorithms.communinty (if needed at all)

# Interactive visualizations in browser     # Check which ones are necessary!!!
from bokeh.io import output_file, show, save, curdoc
from bokeh.layouts import row, column
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, LabelSet, Select, CustomJS
from bokeh.palettes import Blues8, Viridis8
from bokeh.plotting import figure, from_networkx
from bokeh.transform import linear_cmap

# LOGICS:

# Define functions:
# 1. function converting the data to edge dataset -> needs to take clustering criterion from Select as input
# 2. function for creating the network and appending the desired node attributes to its source 
# 3. function for creating a DataTable out of the node source table (the same information as hover)
# 4. function for linear mapping of color and node size (hopefully avoiding the communities?)
# 5. function for plotting the network (shall take size and color from the select)

# Widgets:
# 1. Select: clustering, size, color (on change events)
# 2. DataTable: (update upon hovering)
# 3. 

# IF FANCY: click on the node -> link to the publication

# CODE:

# GLOBAL CONSTANTS
COLUMN_AUTHORS = 'Authors'

# # GLOBAL VARIABLES 
# global plot
# plot = figure()

# FUNCTIONS:

# HELP FUNCTIONS/PROCEDURES
def create_dict_author_attribute (df: pd.DataFrame, col_attribute: str):
    """ Creates a dictionary of authors and the desired attribute (database column) given a df
    
    Parameters
    ----------
    df : pd.Dataframe
                      Source DataFrame storing LIBS publications

    col_attribute   : str
                      criterion for clustering of publications

    Return
    -------
    dict_auth_attr   : dict
                      DataFrame formation of nodes and edges of network graph
    """
    dict_auth_attr = df.set_index(COLUMN_AUTHORS).to_dict()[col_attribute]
    return dict_auth_attr


def add_node_attributes (df: pd.DataFrame, publication_attribute: list, G: nx.Graph):
    """ Creates dictionary from desired attribute and appends it to the nodes
    
    Parameters
    ----------
    df                             : pd.Dataframe
                                     Source DataFrame storing LIBS publications

   

    publication_attributes         : list
                                    list of attributes to be included in the network
                                         
    Return
    -------
    G                               : Networkx graph
                                     Network graph of the publications
    """
    dict_auth_attr = create_dict_author_attribute(df, publication_attribute)
    nx.set_node_attributes(G, name=publication_attribute.lower().replace(" ", "_"), values=dict_auth_attr)  


# MAIN FUNCTIONS
def generate_edges (df: pd.DataFrame, criterion: str):
    """ Uses criterion (taken from a widget) to create an edge dataframe ready for network graphing
    
    Parameters
    ----------
    df : pd.Dataframe
                      Source DataFrame storing LIBS publications

    criterion       : str
                      criterion for clustering of publications

    Return
    -------
    df_edges        : pd.Dataframe
                      DataFrame formation of nodes and edges of network graph
    """
    dict_edges = create_dict_author_attribute(df, criterion)
    df_edges = pd.DataFrame(data = combinations_with_replacement(df[COLUMN_AUTHORS].tolist(), 2), columns = ['Src', 'Dst']) 
    df_edges['Weights'] = (df_edges['Src'].map(dict_edges) == df_edges['Dst'].map(dict_edges)).astype(int)
    df_edges.drop(df_edges.loc[df_edges['Weights']==0].index, inplace = True)
    return df_edges


def create_network (df: pd.DataFrame, df_edges: pd.DataFrame, publication_attributes: list):
    """ Uses criterion (taken from a widget) to create an edge dataframe ready for network graphing
    
    Parameters
    ----------
     df                            : Dataframe
                                    Source DataFrame storing LIBS publications

    df_edges                        : pd.Dataframe
                                    DataFrame ready for formation of nodes and edges of network graph

    publication_attributes          : list
                                    list of attributes to be included in the network
                                         
    Return
    -------
    G                               : graph
                                     Network graph of the publications
    """
    G = nx.from_pandas_edgelist(df_edges, 'Src', 'Dst', 'Weights')
    [add_node_attributes(df, x, G) for x in publication_attributes]
    return G


def plot_network (G: nx.Graph, title: str, node_size_attr: str, node_color_attr: str, color_palette: str):
    """ Uses criterion (taken from a widget) to create an edge dataframe ready for network graphing
    
    Parameters
    ----------
    G                        : Graph
                             Network graph of the publications

    title                    : str
                             title of the plot

    node_size_attr           : str
                             attribute defining the node size  

    node_size_attr           : str
                             attribute defining the node color

    color_palette            : str
                             color palette                            
                                         
    Return
    -------
    plot                     : Figure
                             Network graph of the publications
    """
    #Create a plot
    plot = figure(tooltips = HOVER_TOOLTIPS,
                  tools=HOVER_TOOLS, 
                  active_scroll='wheel_zoom',
                  x_range=Range1d(-1.3, 1.3),
                  y_range=Range1d(-1.3, 1.3),
                  title=title)

    network_graph = from_networkx(G, nx.spring_layout(G, k=0.6, seed = 8), scale=8, center=(0, 0))   # potentially remove scale

    #Set node size and color
    minimum_value_color = min(network_graph.node_renderer.data_source.data[node_color_attr])
    maximum_value_color = max(network_graph.node_renderer.data_source.data[node_color_attr])
    network_graph.node_renderer.glyph = Circle(size=node_size_attr, fill_color=linear_cmap(node_color_attr, color_palette, minimum_value_color, maximum_value_color))
    
    #Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)
    
    #Add Labels
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='10px', background_fill_alpha=.7)
    plot.renderers.append(labels)

    #Add network graph to the plot
    plot.renderers.append(network_graph)

    return plot

def update_plot (attrname, old, new):
    cluster_criterion = select_clustering.value
    plot.title.text = "LIBS publications clustered according to: " + cluster_criterion
    df_LIBS_edges = generate_edges(df_LIBS, cluster_criterion)
    #print(df_LIBS_edges)
    G = create_network(df_LIBS, df_LIBS_edges, NODE_ATTRIBUTES)
    #print(G._node)
    network_graph = from_networkx(G, nx.spring_layout(G, k=0.6, seed = 8), scale=8, center=(0, 0))   # potentially remove scale
    
    minimum_value_color = min(network_graph.node_renderer.data_source.data[color_attribute])
    maximum_value_color = max(network_graph.node_renderer.data_source.data[color_attribute])
    network_graph.node_renderer.glyph = Circle(size=size_attribute, fill_color=linear_cmap(color_attribute, color_palette, minimum_value_color, maximum_value_color))
   
    plot.renderers = []
    plot.renderers.append(network_graph)    
    
    #Add Labels
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='12px', background_fill_alpha=.7)
    plot.renderers.append(labels)
    
#***************
# PROGRAM MAIN
#***************

# Constants
NODE_ATTRIBUTES = ['Citationsadj', 'Year', 'Application', 'Field', 'Classes', 'Preprocessing',
                   'Spectral region', 'Classification', 'Classifiers tried', 'Classification details',
                   'Model validation', 'Validation internal']

HOVER_TOOLTIPS = [("Publication", "@index"), 
                  ("Application", "@application"), 
                  ("Samples", "@field"), 
                  ("Classes", "@classes"), 
                  ("Preprocessing", "@preprocessing"), 
                  ("Spectral input", "@spectral_region"), 
                  ("Classifiers employed", "@classifiers_tried"), 
                  ("Best model", "@classification_details"),
                  ("Validation", "@model_validation"),
                  ("Performance", "@validation_internal")
                  ]

HOVER_TOOLS = ["pan,wheel_zoom,save,reset, lasso_select"]

SEED = 8

# automatized:  HOVER_TOOLTIPS 
# [("Publication", "@index")] + [(x, '@'+x.lower()) for x in NODE_ATTRIBUTES if x != 'Citationsadj']
plot_title = 'LIBS classification publications'
cluster_criterion = 'Year'
size_attribute = 'citationsadj'
color_attribute = 'year'
color_palette = Blues8

# Displayed widget values 
select_clustering = Select(value=cluster_criterion, title='Cluster according to:', options=[x for x in NODE_ATTRIBUTES if x != 'Citationsadj'])
select_sizing = Select(value=size_attribute, title='Node size according to:', options=['Citationsadj'])
select_coloring = Select(value=color_attribute, title='Node color according to:', options=['Year'])

# Load the data # google how to make this universal
# df = pd.read_csv(join(dirname(__file__)), 'data\LIBS_overview.csv.csv')   
df_LIBS = pd.read_csv(join(dirname(__file__),'LIBS_overview_exp.csv'), delimiter=';') 

# Cluster publications according to the select -> !!! add select as input argument
df_LIBS_edges = generate_edges(df_LIBS, cluster_criterion)

# Create the network 
G = create_network(df_LIBS, df_LIBS_edges, NODE_ATTRIBUTES)

# Create table from the node dictionary :)!!!!

# Plot the network
plot = plot_network(G, plot_title, size_attribute, color_attribute, color_palette)

select_clustering.on_change('value', update_plot)

# Update the value on change
# select_clustering.js_on_change("value", CustomJS(code="""
#     console.log('select: value=' + this.value, this.toString())
# """))

controls = column(select_clustering, select_sizing, select_coloring)

curdoc().add_root(row(plot, controls))
curdoc().title = "LIBS"

#show(row(plot, controls))
#output_file('test.html')
#save(plot, filename=f"{plot_title}.html")
