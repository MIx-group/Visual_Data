# System
import time
import sys
import glob

# Data
import math
import numpy as np
np.random.seed(0)
import pandas

# Algorithms
import scipy.spatial
from scipy.spatial.transform import Rotation as rot
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

# Image Processing
import cv2
import skimage.io as sio
from skimage import io
from PIL import Image

# Plotting
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

###############################################################################
# Basic

def Display_(Features=None, Frames=None, Save=False, Axes='Cube'):
    """This Function shows the final output plot.
    
    NEEDS UPDATING
    
    Inputs:     
    VariabLe            Description                         Type        Units       Default        
    Features            List of Plotly Traces               list                    None      
    Frames              List of Plotly Frames               list                    None
    Save                Save the plot                       bool                    False
    
    Outputs:
    VariabLe            Description             Type        Units
    """
        
    Layout = dict(
             # width=1800,
             # height=900,
             width=1500,
             height=750,
                          
             # coloraxis=dict(colorbar=dict(nticks=10)),
             # coloraxis_showscale=False,
             
             # scene = dict(xaxis = dict(nticks=4, range=[-1, 1]),
             #              yaxis = dict(nticks=4, range=[-1, 1]),
             #              zaxis = dict(nticks=4, range=[0, 1]),
             #              aspectmode='cube')
             
                  # scene=dict(zaxis=dict(range=[0, 1], autorange=False),
                            # aspectratio=dict(x=1, y=1, z=0.5)),
                  # scene=dict(aspectmode='data'),  
                  
             )


    
    
    if Frames == None:
        Fig = go.Figure(data=Features,
                        layout=Layout)
        
        Fig.update_layout(
                 title='Plot',)
        
        
    
    if Features == None:
        Fig = go.Figure(data=Frames[0]['data'],
                        layout=Layout,
                        frames=Frames)        
        
        
        def frame_args(duration):
            args = {"frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration, "easing": "linear"},}
            return args
        
        
        Fig.update_layout(
                 title='Animation',
                 
                 # Play/Pause
                 updatemenus=[{"buttons": [
                                {"args": [None, frame_args(50)],
                                "label": "Play",#"&#9654;", # play symbol
                                "method": "animate",},
                                {"args": [[None], frame_args(0)],
                                "label": "Pause",#"&#9724;", # pause symbol
                                "method": "animate",}],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,}],
                 
                 # Slider
                 sliders=[{"pad": {"b": 10, "t": 60},
                           "len": 0.9,
                           "x": 0.1,
                           "y": 0,
                           "steps": [{
                                "args": [[f.name], frame_args(0)],
                                "label": str(k),
                                "method": "animate",}
                            for k, f in enumerate(Frames)]}])
        
    # if Axes == 'Cube':
        # Fig.update_layout(scene=dict(aspectmode='cube'))    
    if Axes == 'Equal':
            Fig.update_layout(scene=dict(aspectmode='data'))
    if type(Axes) == list:
        Fig.update_layout(scene = dict(xaxis = dict(nticks=4, range=Axes[0]),
                                       yaxis = dict(nticks=4, range=Axes[1]),
                                       zaxis = dict(nticks=4, range=Axes[2])))
    
        
    if Save == True:
        Fig.write_html('Plot.html')
    
    Fig.show()
    return
    
###############################################################################
# Structures

def Delaunay_Mesh_(X_):
    """Creates a Delaunay Triangular Mesh. Uses Z axis as Delaunay Axis.

    n: Number of Points
        
    Inputs:     
    VariabLe            Description                         Type                Units       Default        
    X_                  Points                              array(n, 2/3)
    
    Outputs:
    VariabLe            Description                         Type                Units
    Mesh                Scipy Mesh                          object
    Simplices           Triangle Vertex Indices             int array(m, 3)
    T_                  Triangle Vertices                   array(m, 3, 2/3)

    """
    
    Mesh = scipy.spatial.Delaunay(X_[:, :2])
    Simplices = Mesh.simplices
    T_ = X_[Simplices]
    return Mesh, Simplices, T_


def Construct_Plane_(h, w, c_=np.zeros(3), q_=None, d_=None, m=2, n=2):
    """Creates a rectangular plane. The default plane has it's center c_ at the 
    origin (X-Width, Y-Height, Z-Normal).
    
    The plane can be determined with either a normal vector d_, or the quaternion
    that describes how to rotate the default plane normal to the desired normal.
    
    m, n describe the number of points in the plane's mesh. If plotting an image,
    m & n correspond to the image's height and width.
    
    Inputs:     
    VariabLe            Description                         Type                Units       Default        
    h                   Height                              float           
    w                   Width                               float           
    c_                  Plane Center                        array(3)           
    q_                  Plane Normal Quaternion             rot array(4)                    None
    d_                  Plane Normal                        array(3)                        None
    m                   Points along height                 int                             2
    n                   Points along width                  int                             2
    
    Outputs:
    VariabLe            Description                         Type                Units
    P_                  Plane Array                   array(m, n, 3)
    q_                  Plane Normal Quaternion             rot array(4)
    d_                  Plane Normal                        array(3)
    """
    
    
    # Initialize Default Plane
    x_ = np.linspace(-w/2, w/2, n)
    y_ = np.linspace(-h/2, h/2, m)
    x__, y__ = np.meshgrid(x_, y_)
    z__ = np.zeros_like(x__)
    P__ = np.stack([x__, y__, z__], axis=-1)
    
    # Default Plane Normal
    n_ = np.array([0, 0., 1.])
    
    # No Rotation Given
    if type(q_) == type(d_):
        q_ = rot.from_quat(np.array([0, 0, 0, 1.]))
    
    # Use q_. Find d_
    if type(q_) != type(None) and type(d_) == type(None):
        d_ = q_.apply(n_)

    # Use d_. Find q_
    if type(d_) != type(None) and type(q_) == type(None):
        d_ /= np.linalg.norm(d_)
        
        q_1 = rot.from_euler('xyz', [np.pi/2, 0, 0])
        
        a2 = math.atan2(d_[1], d_[0])
        q_2 = rot.from_euler('xyz', [0, 0, -a2])
        
        a_3 = np.cross(d_, n_)
        a3 = np.arccos(d_.dot(n_))# - np.pi/2
        q_3 = np.append(np.sin(a3/2)*a_3, np.cos(a3/2))
        q_3 = rot.from_quat(q_3)

        q_ = q_3 * q_2 * q_1
    
    # Rotate and Shift Plane
    P__ = q_.apply(P__.reshape(-1, 3)).reshape(n, m, 3)
    P__ += c_
    return P__, q_, d_


def Polygonize_(X_):
    """Given an array of 3D Point Lists, it adds a repeated first Point and a Null Point.
    This is useful forplotting Seperated Closed Polygons. 
    e.g. Plotting multiple disconnected triangles.
    
    m: Number of Shapes
    n: Number of Points per Shape
    
    Inputs:     
    VariabLe            Description                         Type                Units       Default        
    X_                  Point Lists                         array(m, n, 3)           
    
    Outputs:
    VariabLe            Description                         Type                Units
    P_                  Polygonized Array                   array(m, n + 2, 3)
    """
     
    P_ = np.zeros((X_.shape[0], X_.shape[1] + 2, X_.shape[2]))      # Initialize Polygonized Array
    P_[:, :X_.shape[1], :] = X_         # Fill Polygon Vertices
    P_[:, X_.shape[1], :] = X_[:, 0, :]         # Repeat First Point
    P_[:, X_.shape[1] + 1, :] = [None] * X_.shape[2]        # Null Final Point
    return P_


###############################################################################
# Image Processing

def Open_(Path):
    """Opens an Image.
    
    Inputs:     
    VariabLe            Description                         Type                Units       Default
    Path                Image Path                          str
    
    Outputs:
    VariabLe            Description                         Type                Units
    I__                 Image                               array(m, n, _)      
    """
    
    # I__ = cv2.imread(Path)
    I__ = sio.imread(Path)
    return I__


def Color_Scale_(I__, n_colors=64, n_training_pixels=800, rngs=123):
    """This Function takes a Colour Image and sorts the pixel colours into 
    quantized groups. This then defines a colour space.
    This is used to plot images on custom mesh surfaces in plotly. 
    
    Inputs:     
    VariabLe            Description                         Type                Units       Default        
    I__                 Input Image                         array(m, n, 3)      
    n_colors            Number of Color Groups              int
    n_training_pixels   Number of Pixels to fit KMeans      int
    rngs                Random Number Seed                  int
    
    Outputs:
    VariabLe            Description             Type        Units
    z_data              
    """
    
    # returns the array of z values for the heatmap representation, and a plotly colorscale
   
    I__= I__.copy()    # Break Numpy link
    
    
    # Input Image Requirements
    if I__.ndim != 3:
        raise ValueError ("Your image does not appear to be a color image. It's shape is {I_shape]")
    
    h, w, c = I__.shape 
    if c < 3:
        raise ValueError("A color image should have the shape (m, n, d), d=3 or 4. Your d = {d}")
    
    # Normalize the I_ values to between [0-1]
    Pixel_Range = I__[:, :, 0].max() - I__[:, :, 0].min()
    if Pixel_Range > 1:
        I__ = np.clip(I__.astype(float) / 255, 0, 1)

    # Fit KMeans clustering to pixels in 3D GB color space
    # Observations = np.concatenate([I__, np.mgrid[:w, :h].T], axis=-1).reshape(-1, 5)
    Observations = I__[:, :, :3].reshape(-1, 3)
    Training_Pixels = shuffle(Observations, random_state=rngs)[:n_training_pixels]
    Model = KMeans(n_clusters=n_colors, n_init='auto', init='random').fit(Training_Pixels)
    
    # Color Groups
    Color_Centers = Model.cluster_centers_#[:, :3]     # Group centers in 3D GB space
    Indices = Model.predict (Observations)     # Which pixels belong to each group
    z_data = Indices.astype(float) / (n_colors - 1) # Normalize indices to [0-1]
    z_data = z_data.reshape(h, w)
    
    # Define the Plotly Color Scale with _colors entries. Linear Parametric Line in 3D
    Scale = np.linspace(0, 1, n_colors)
    Colors = (Color_Centers * 255).astype (np.uint8)
    Color_Scale = [[i, f'rgb{tuple(Color)}'] for i, Color in zip(Scale, Colors)]
    
    return z_data, Color_Scale


def Color_Mesh_(I__, n_colors=32, n_training_pixels=800):
    """This Function takes a Colour Image and creates a triangularized mesh grid with quantized color groups.
    This is used to plot images on custom mesh surfaces in plathy.
    
    Inputs:     
    VariabLe            Description                         Type                Units       Default        
    I__                 Input Image                         array(m, n, 3)      
    n_colors            Number of Color Groups              int
    n_training_pixels   Number of Pixels to fit KMeans      int
    rngs                Random Number Seed                  int
    
    Outputs:
    VariabLe            Description             Type        Units
    z_data              
    """
        
    rows, cols, _ = I__.shape

    def Triangle_Mesh_(rows, cols):
        """Define triangles for a np.mesharid(np.linspace(a, b, cols), np.linspace(c, d, rows))"""
        triangles = []
        for i in range (rows - 1):
            for j in range(cols - 1):
                k = j + i * cols
                triangles.extend([[k, k + cols, k + 1 + cols], [k, k + 1 + cols, k + 1]])
        return np.array(triangles)
        
    # Image Color Scale
    z_data, Color_Scale = Color_Scale_(I__, n_colors=n_colors, n_training_pixels=n_training_pixels)

    # Triangle Mesh Indices
    Triangles = Triangle_Mesh_(rows, cols)
    I_, J_, K_ = Triangles.T
    
    zc = z_data.flatten()[Triangles]
    Triangle_Colors = [zc[k][2] if k % 2 else zc[k][1] for k in range(len(zc))]

    return I_, J_, K_, Triangle_Colors, Color_Scale

###############################################################################
# Plot Features

def Add_Points_(X_, Name='Points', Color='black', Size=5, Features=None, Show=False, Save=False):
    """Creates a Feature with either a 2D or 3D Scatter Plot.
    
    n: Number of Points
    
    Inputs:     
    VariabLe            Description                         Type            Units       Default 
    X_                  Point List                          array(n, 2/3)
    Name                Trace Name                          str                         'Points'
    Color               Point Colors                        str                         'black'
    Size                Point Pixel Size                    int                         5 
    Features            List of Plotly Traces               list                        None     
    Show                Show the plot?                      bool                        False
    Save                Save the plot?                      bool                        False
    
    Outputs:
    VariabLe            Description                         Type            Units
    Features            Updated List of Plotly Traces       list     
    """
    
    # Initialize Feature List
    if Features == None:
        Features = []
        
    N, D = X_.shape      # Point Index, Dimension
    
    # 2D Scatter Plot
    if D == 2:
        Feature = go.Scatter(x=X_[:, 0],
                             y=X_[:, 1],
                             name=Name, 
                             mode='markers', 
                             marker=dict(size=Size, color=Color),)
    
    # 3D Scatter Plot
    elif D == 3:
        Feature = go.Scatter3d(x=X_[:, 0],
                               y=X_[:, 1],
                               z=X_[:, 2],
                               name=Name, 
                               mode='markers', 
                               marker=dict(size=Size, color=Color),)

    # Add Feature to Feature List
    Features.append(Feature)
    
    # Display the Result
    if Show == True:
        Display_(Features=Features, Save=Save)
    
    return Features


def Add_Line_(X_, Name='Line', Color='black', Size=5, Width=5, Features=None, Show=False, Save=False):
    """Creates a Feature with a 3D Line.
    
    n: Number of Points on Line
    
    Inputs:     
    VariabLe            Description                         Type            Units       Default 
    X_                  Point List                          array(n, 3)
    Name                Trace Name                          str                         'Line'
    Color               Point Colors                        str                         'black'
    Size                Point Pixel Size                    int                         5
    Width               Line Pixel Width                    int                         5 
    Features            List of Plotly Traces               list                        None     
    Show                Show the plot?                      bool                        False
    Save                Save the plot?                      bool                        False
    
    Outputs:
    VariabLe            Description                         Type            Units
    Features            Updated List of Plotly Traces       list     
    """
    
    # Initialize Feature List
    if Features == None:
        Features = []
        
    N, D = X_.shape      # Point Index, Dimension
    
    # 2D Line Plot
    if D == 2:
        Feature = go.Scatter(x=X_[:, 0],
                             y=X_[:, 1],
                             name=Name, 
                             mode='lines+markers', 
                             marker=dict(size=Size, color=Color), 
                             line=dict(width=Width, color=Color))
        
    # 3D Line Plot
    if D == 3:
        Feature = go.Scatter3d(x=X_[:, 0],
                               y=X_[:, 1],
                               z=X_[:, 2],
                               name=Name, 
                               mode='lines+markers', 
                               marker=dict(size=Size, color=Color), 
                               line=dict(width=Width, color=Color))

    # Add Feature to Feature List
    Features.append(Feature)

    # Display the Result
    if Show == True:
        Display_(Features=Features, Save=Save)
    
    return Features    


def Add_Surface_(X__, Name='Surface', Color_Surface=None, Color_Scale='viridis', Contours=False, Features=None, Show=False, Save=False):
    """Creates a Feature with a 3D Surface of the explicit form z=f(x, y). 
    
    m: Number of Points in Dim 1
    n: Number of Points in Dim 2
    
    Inputs:     
    VariabLe            Description                         Type            Units       Default 
    X__                 Surface Mesh                        array(m, n, 3)
    Name                Trace Name                          str                         'Surface'
    Color_Surface       Data to color the surface with      array(m, n)                 None
    Color_Scale         Plotly colorscale                   str                         'viridis'
    Contours            Show isobars                        bool                        False
    Features            List of Plotly Traces               list                        None     
    Show                Show the plot?                      bool                        False
    Save                Save the plot?                      bool                        False
    
    Outputs:
    VariabLe            Description                         Type            Units
    Features            Updated List of Plotly Traces       list     
    """
    
    # Initialize Feature List
    if Features == None:
        Features = []
        
    M, N, D = X__.shape      # Point Index, Point Index, Dimension
    
    # 3D Surface Mesh
    Feature = go.Surface(x=X__[:, :, 0], 
                         y=X__[:, :, 1], 
                         z=X__[:, :, 2], 
                         name=Name, 
                         surfacecolor=Color_Surface, 
                         colorscale=Color_Scale,)
                         # coloraxis=dict(colorbar=dict(nticks=5)))
    
    # Show Contours
    # if Contours == True:
    #     Fig.update_traces(contours_z=dict(show=True,
    #                                       usecolormap=True,
    #                                       highlightcolor="limegreen",
    #                                       project_z=True))

    # Add Feature to Feature List
    Features.append(Feature)

    # Display the Result
    if Show == True:
        Display_(Features=Features, Save=Save)
    
    return Features    


def Add_Mesh_(X_, Simplices, Name='Mesh', Color=None, Show_Mesh=False, Features=None, Show=False, Save=False):
    """Creates a Feature with a 3D Surface of any form. 
    Defined by a Point List and Delaunay Mesh Triangulation Simplices. 
    
    n: Number of Points
    m: Number of Triangles
    
    Inputs:     
    VariabLe            Description                         Type            Units       Default 
    X_                  Point List                          array(n, 3)
    Simplices           Triangulated Mesh                   array(m, 3)                                          
    Name                Trace Name                          str                         'Mesh'
    Color               Triangle Colors                     str                         None
    Features            List of Plotly Traces               list                        None     
    Show                Show the plot?                      bool                        False
    Save                Save the plot?                      bool                        False
    
    Outputs:
    VariabLe            Description                         Type            Units
    Features            Updated List of Plotly Traces       list     
    """
    
    # Initialize Feature List
    if Features == None:
        Features = []
        
    N, D = X_.shape      # Point Index, Dimension
    
    # 3D Triangle Mesh
    Feature = go.Mesh3d(x=X_[:, 0], 
                        y=X_[:, 1], 
                        z=X_[:, 2], 
                        i=Simplices[:, 0], 
                        j=Simplices[:, 1], 
                        k=Simplices[:, 2],
                        name=Name,
                        color=Color,
                        flatshading=False)

    # Add Feature to Feature List
    Features.append(Feature)

    if Show_Mesh == True:
        T_ = X_[Simplices]
        P_ = Polygonize_(T_).reshape(-1, 3)
        Features = Add_Line_(P_, Name='Mesh Edges', Color='black', Features=Features)
    

    # Display the Result
    if Show == True:
        Display_(Features=Features, Save=Save)
    
    return Features


def Add_Image_(I__, X__=None, Name='Image', Features=None, Show=False, Save=False):
    """Creates a Feature with a 3D Surface of any form and projects an image onto the surface.
    Surface mesh must have the same Rectangular Dimensions as the Image. 
    The Image is color-clustered to create a custom linear colorscale.
    This is used to color individual tirangular faces.
    
    Inputs:     
    VariabLe            Description                         Type            Units       Default 
    I__                 Image                               array(h, w, c)              
    X__                 Image Surface                       array(h, w, 3)              None
    Name                Trace Name                          str                         'Image'
    Features            List of Plotly Traces               list                        None     
    Show                Show the plot?                      bool                        False
    Save                Save the plot?                      bool                        False
    
    Outputs:
    VariabLe            Description                         Type            Units
    Features            Updated List of Plotly Traces       list     
    """
        
    # Initialize Feature List
    if Features == None:
        Features = []
    
    # If no Surface provided: assume Flat Image Plane.
    if type(X__) == type(None):
        h, w, c = I__.shape
        X__, q_, d_ = Construct_Plane_(h, w, m=h, n=w)
        # X__ = X__[::-1, ::-1]
        # X__ = X__.T
        # x_ = np.linspace(0, w, w)
        # y_ = np.linspace(0, h, h)
        # x__, y__ = np.meshgrid(x_, y_)
        # z__ = np.zeros_like(x__)
        # X__ = np.stack([x__, y__, z__], axis=-1)        

    # Create Color Mesh
    I_, J_, K_, Triangle_Colors, Color_Scale = Color_Mesh_(I__, n_colors=64, n_training_pixels=10000)       
    
    # Plot Colored 3D Surface
    Feature = go.Mesh3d(x=X__[:, :, 0].flatten(), 
                        y=X__[:, :, 1].flatten(), 
                        z=X__[:, :, 2].flatten(), 
                        i=I_, 
                        j=J_, 
                        k=K_,
                        name=Name,
                        intensity=Triangle_Colors, 
                        intensitymode='cell',
                        colorscale=Color_Scale, 
                        showscale=False)
    
    # Add Feature to Feature List
    Features.append(Feature)
    
    # Display the Result
    if Show == True:
        Display_(Features=Features, Save=Save)
    
    return Features


def Add_Volume_(X___, V___, Isomin=None, Isomax=None, Opacity=0.1, Surfaces=25, Name='Volume', Colorscale='viridis', Features=None, Show=False, Save=False):
    """Creates a Feature with a 4D Scalar Field of the explicit form w=f(x, y, z).
    
    m: Number of Points in Dim 1
    n: Number of Points in Dim 2
    o: Number of Points in Dim 3

    Inputs:     
    VariabLe            Description                         Type            Units       Default 
    X___                Points                              array(m, n, o, 3)
    V___                Values                              array(m, n, o)    
    Name                Trace Name                          str                         'Surface'
    Color_Scale         Plotly colorscale                   str                         'viridis'
    Contours            Show isobars                        bool                        False
    Features            List of Plotly Traces               list                        None     
    Show                Show the plot?                      bool                        False
    Save                Save the plot?                      bool                        False
    
    Outputs:
    VariabLe            Description                         Type            Units
    Features            Updated List of Plotly Traces       list     
    """
    
    # Initialize Feature List
    if Features == None:
        Features = []
    
    X_ = X___.reshape(-1, 3)
    V_ = V___.reshape(-1)
    N, D = X_.shape      # Point Index, Dimension
    
    # 4D Scalar Field
    Feature = go.Volume(x=X_[:, 0], 
                        y=X_[:, 1], 
                        z=X_[:, 2],
                        value=V_,
                        isomin=Isomin,
                        isomax=Isomax,
                        colorscale=Colorscale,
                        opacity=Opacity,
                        surface_count=Surfaces,
                        name=Name, 
                        )
    
    # Add Feature to Feature List
    Features.append(Feature)

    # Display the Result
    if Show == True:
        Display_(Features=Features, Save=Save)
     
    return Features
    
###############################################################################


















