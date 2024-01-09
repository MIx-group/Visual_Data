from Plotly_Functions import *

# This code demonstrates how to use the plotting functions in Plotly_Functions.py
# See Plotly_Functions for deeper explantion of functions and which inputs are optional

###############################################################################
Features = []   # Features are accumulated in this list

N = 50
D = 3

###############################################################################
# Unconnected Points
X_ = np.random.random(N*D).reshape(N, D)
print('Scatter Points: ', X_.shape)
Features = Add_Points_(X_, Name='Points', Color='black', Size=5, Features=Features, Show=False, Save=False)

###############################################################################
# Points Connected in Line
X_ = np.random.random(N*D).reshape(N, D) + np.array([1, 0, 0])
print('Line Points: ', X_.shape)
Features = Add_Line_(X_, Name='Line', Color='red', Size=5, Width=5, Features=Features, Show=False, Save=False)

###############################################################################
# Rectangular Surface
X__ = np.zeros((N, N, D))
X__[:, :, :2] = np.mgrid[:N, :N].T / (N-1)
X__[:, :, 2] = X__[:, :, 0]**2
X__ += np.array([[2, 0, 0]])
print('Surface Points: ', X__.shape)
Features = Add_Surface_(X__, Name='Surface', Color_Surface=X__[:, :, 2], Color_Scale='viridis', Contours=False, Features=Features, Show=False, Save=False)

###############################################################################
# Custom Mesh Surface
X_ = np.zeros((N, D))
X_[:, :2] = np.random.random(N*2).reshape(N, 2)
X_[:, 2] = X_[:, 0]**2
X_ += np.array([3, 0, 0])

Mesh, Simplcies, T_ = Delaunay_Mesh_(X_)
P_ = Polygonize_(T_)

print('Mesh Points: ', X_.shape)
print('Mesh Triangles: ', T_.shape)
Features = Add_Mesh_(X_, Mesh.simplices, Name='Mesh', Color='green', Show_Mesh=True, Features=Features, Show=False, Save=False)

###############################################################################
# Image Surface

# Load Image
I__ = Open_('MIX.png')[::10, ::10]
h, w, c = I__.shape
plt.imshow(I__), plt.show()

# Image Surface
q_ = rot.from_quat([0, np.sin(np.pi/2), 0, np.cos(np.pi/2)])
P__, q_, d_ = Construct_Plane_(h, w, q_=q_, m=h, n=w)
P__ /= w*0.2
P__ += np.array([[2.5, 0.5, -0.5]])

Features = Add_Image_(I__, X__=P__, Name='Image', Features=Features, Show=False, Save=False)

###############################################################################
# Volume

X___ = np.mgrid[:N, :N, :N].T / (N-1)
V___ = np.linalg.norm(np.array([1, 1, 0]) - X___, axis=-1)
X___ += np.array([4, 0, 0])

Features = Add_Volume_(X___, V___, Isomin=None, Isomax=1, Opacity=0.1, Surfaces=25, Name='Volume', Colorscale='viridis', Features=Features, Show=False, Save=False)

###############################################################################

Display_(Features, Axes='Equal', Save=True)









