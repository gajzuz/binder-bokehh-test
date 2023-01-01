# Library imports

# System paths
from os.path import join, dirname

# Data wrangling
from scipy import mean
import pandas as pd

# Network graph 
from itertools import combinations_with_replacement     
import networkx as nx                                  

# Interactive visualizations in browser     
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, LabelSet, Select, ColorBar, CustomJS
from bokeh.palettes import Blues8
from bokeh.plotting import figure, from_networkx
from bokeh.transform import linear_cmap

# GLOBAL CONSTANTS
COLUMN_AUTHORS = 'Authors'

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
   

def allocate_clusters (label_cluster: str, x: list, y:list, cluster_labels_all:list):
    indices = [i for i in range(len(cluster_labels_all)) if cluster_labels_all[i]==label_cluster]
    range_x = abs(abs(min([x[i] for i in indices])) - abs(max([x[i] for i in indices])))
    if len(indices) == 1:
        cluster_x = mean([x[i] for i in indices])-0.05
    else:
        cluster_x = mean([x[i] for i in indices])-range_x
    range_y = abs(abs(min([y[i] for i in indices])) - abs(max([y[i] for i in indices])))
    if len(indices) == 1:
        cluster_y = mean([y[i] for i in indices])+0.05
    else:
        cluster_y = mean([y[i] for i in indices])+range_y
    return [label_cluster, cluster_x, cluster_y]    
    

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


def plot_network (G: nx.Graph, title: str, node_size: int, node_color_attr: str, color_palette: str):
    """ Uses criterion (taken from a widget) to create an edge dataframe ready for network graphing
    
    Parameters
    ----------
    G                        : Graph
                             Network graph of the publications

    title                    : str
                             title of the plot

    node_size           : int
                             attribute defining the node size  

    node_color           : str
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
                  x_range=Range1d(-3, 3),
                  y_range=Range1d(-5, 1.5),
                  sizing_mode = 'scale_both',
                  title=title)

    network_graph = from_networkx(G, nx.fruchterman_reingold_layout(G, seed = 8), scale=8, center=(0, 0))   # potentially remove scale

    #Set node size and color
    minimum_value_color = min(network_graph.node_renderer.data_source.data[node_color_attr])
    maximum_value_color = max(network_graph.node_renderer.data_source.data[node_color_attr])
    network_graph.node_renderer.glyph = Circle(size=node_size, fill_color=linear_cmap(node_color_attr, color_palette, minimum_value_color, maximum_value_color))
    
    #Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.3, line_width=1)
    
    #Add publication labels
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='12px', background_fill_alpha=.7)
    #plot.renderers.append(labels)
    
    #Add cluster labels
    cluster_labels_all = network_graph.node_renderer.data_source.data[cluster_criterion.lower().replace(" ", "_")]
    cluster_labels = list(set(cluster_labels_all))
    list_clusters = [allocate_clusters(i, x, y, cluster_labels_all) for i in cluster_labels]
    cluster_names = [i[0] for i in list_clusters]
    cluster_xs = [i[1] for i in list_clusters]
    cluster_ys = [i[2] for i in list_clusters]
    cluster_source = ColumnDataSource({'x': cluster_xs, 'y': cluster_ys, 'name': cluster_names})
    cluster_labelset = LabelSet(x='x', y='y', text='name', source=cluster_source, background_fill_color='white', text_font_size='20px', background_fill_alpha=0)
    # plot.renderers.append(cluster_labelset)

    #Add network graph to the plot
    #plot.renderers.append(network_graph)

    #Plot all
    #plot.renderers = []
    plot.renderers.append(network_graph) 
    plot.renderers.append(labels)
    plot.renderers.append(cluster_labelset)
    
    return plot

def update_plot (attrname, old, new):
    cluster_criterion = select_clustering.value
    plot.title.text = "LIBS publications clustered according to: " + cluster_criterion
    
    #pack in a function plot_callback
    df_LIBS_edges = generate_edges(df_LIBS, cluster_criterion)
    G = create_network(df_LIBS, df_LIBS_edges, NODE_ATTRIBUTES)
    network_graph = from_networkx(G, nx.fruchterman_reingold_layout(G, seed = 8), scale=8, center=(0, 0))   # potentially remove scale
    minimum_value_color = min(network_graph.node_renderer.data_source.data[color_attribute])
    maximum_value_color = max(network_graph.node_renderer.data_source.data[color_attribute])
    network_graph.node_renderer.glyph = Circle(size=node_size, fill_color=linear_cmap(color_attribute, color_palette, minimum_value_color, maximum_value_color))
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.3, line_width=1)
       
    # Add Labels
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='12px', background_fill_alpha=.7)
    
    # Add cluster labels
    cluster_labels_all = network_graph.node_renderer.data_source.data[cluster_criterion.lower().replace(" ", "_")]
    cluster_labels = list(set(cluster_labels_all))
    list_clusters = [allocate_clusters(i, x, y, cluster_labels_all) for i in cluster_labels]
    cluster_names = [i[0] for i in list_clusters]
    cluster_xs = [i[1] for i in list_clusters]
    cluster_ys = [i[2] for i in list_clusters]
    cluster_source = ColumnDataSource({'x': cluster_xs, 'y': cluster_ys, 'name': cluster_names})
    cluster_labelset = LabelSet(x='x', y='y', text='name', source=cluster_source, background_fill_color='white', text_font_size='20px', background_fill_alpha=0)
    
    # Plot all
    plot.renderers = []
    plot.renderers.append(network_graph) 
    plot.renderers.append(labels)
    plot.renderers.append(cluster_labelset)
    
#***************
# PROGRAM MAIN
#***************

# Constants
NODE_ATTRIBUTES = ['Year', 'Journal', 'Application', 'Materials', 'Number of classes', 'Classes', 'Preprocessing',
                   'Feature selection', 'Spectral input', 'Classification approaches', 'Classification',
                   'Best model', 'Validation internal', 'Validation external', 'Software']

CLUSTERING_CRITERIONS = ['Application', 'Classification', 'Software']

HOVER_TOOLTIPS = [("Publication", "@index")] + [(x, '@'+x.lower().replace(" ", "_")) for x in NODE_ATTRIBUTES if x != 'Year']

HOVER_TOOLS = ["pan, box_zoom, wheel_zoom,save,reset"]

SEED = 8

plot_title = 'LIBS classification publications'
cluster_criterion = 'Application'
node_size = 15
color_attribute = 'year'
color_palette = Blues8

# Displayed widget values 
select_clustering = Select(value=cluster_criterion, title='Cluster according to:', options=[x.replace('_', ' ') for x in NODE_ATTRIBUTES])
#select_sizing = Select(value=node_size, title='Node size according to:', options=['Constant'])
#select_coloring = Select(value=color_attribute, title='Node color according to:', options=['Year'])

# Load the data
df_LIBS = pd.read_csv(join(dirname(__file__),'LIBS_overview_final.csv'), delimiter=';') 

# Cluster publications according to the select -> !!! add select as input argument
df_LIBS_edges = generate_edges(df_LIBS, cluster_criterion)

# Create the network 
G = create_network(df_LIBS, df_LIBS_edges, NODE_ATTRIBUTES)

# Plot the network
plot = plot_network(G, plot_title, node_size, color_attribute, color_palette)

select_clustering.on_change('value', update_plot)

# Update the value on change
# select_clustering.js_on_change("value", CustomJS(code="""
#     console.log('select: value=' + this.value, this.toString())
# """))

controls = column(select_clustering)

curdoc().add_root(row(plot, controls))
curdoc().title = "LIBS"
