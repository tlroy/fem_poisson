# Simple FEM implementation for Poisson's equation
import numpy as np
from math import sin,pi

# define source term
def f(x):
    return 4*(-x[1]**2 + x[1])*sin(pi*x[0])

## Create mesh ##

Nx = 40 # vertices in x-direction
Ny = 50 # vertices in y-direction
N = Nx*Ny # number of vertices
Ne = (Nx-1)*(Ny-1) # number of elements
hx = 1./(Nx-1) # grid size in x-direction
hy = 1./(Ny-1) # grid size in y-direction
Nd = 2*(Nx-1) + 2*(Ny-1) # number of Dirichlet edges

# Coordinates of different vertices
coordinates = np.empty((N,2)) # each line: [x-coord, y-coord]
k = 0
for j in range(0,Ny):
    for i in range(0,Nx):
        coordinates[k,:] = [i*hx,j*hy]
        k += 1

# Vertices in each element
elements = np.empty((Ne,4), dtype='int') # each line: [vertex1 #, vertex2 #, vertex3 #, vertex4 #]
k = 0
for j in range(0,Ny-1):
    for i in range(0,Nx-1):
        elements[k,:] = [i+j*Nx, i+1+j*Nx, (j+1)*Nx+i+1, (j+1)*Nx+i]
        k += 1

# Edges on boundary        
dirichlet = np.empty((Nd,2), dtype='int') # each line: [vertex1 #, vertex2 #] 
for i in range(0, Nx-1):
    dirichlet[i,:] = [i, i+1]
    dirichlet[Nx-1+i,:] = [N-Nx+i,N-Nx+1+i]
for i in range(0, Ny-1):
    dirichlet[Nd-2*(Ny-1)+i,:] = [i*Nx, (i+1)*Nx]
    dirichlet[Nd-(Ny-1)+i,:] = [(i+1)*Nx-1, (i+2)*Nx-1]
dirichlet_nodes = np.unique(dirichlet)
   
## Finite element method ##   
   
# Local stiffness matrix
detJ = hx*hy # determinant of Jacobian of transformation [[hx,0],[0,hy]]
D = np.array([[2,-2],[-2,2]])/hx**2 + np.array([[2,1],[1,2]])/hy**2
O = np.array([[-1,1],[1,-1]])/hx**2 + np.array([[-1,-2],[-2,-1]])/hy**2
M1 = np.hstack((D,O))
M2 = np.hstack((O,D))
M = np.concatenate((M1,M2))
M = detJ/6*M # local stiffness matrix
    
# Assemble global stiffness matrix
A = np.zeros((N,N)) #should use sparse matrix, but this is simpler for now
for i in range(0,Ne):
    A[np.ix_(elements[i,:],elements[i,:])] = A[np.ix_(elements[i,:],elements[i,:])] +  M

# Assemble right-hand side
b = np.zeros((N,1))
for i in range(0,Ne):
    centre = sum(coordinates[elements[i,:]])/4
    b[elements[i,:]] = b[elements[i,:]] + detJ/4*f(centre)

# Initialise solution vector
u = np.zeros((N,1))
# Fix Dirichlet nodes. Zero-Dirichlet so don't need to modify b
free_nodes = list(set(range(0,N)) - set(dirichlet_nodes))
# Solve inner problem
u[free_nodes] = np.linalg.solve(A[np.ix_(free_nodes,free_nodes)], b[free_nodes])


## Visualisation ##
import matplotlib.pyplot as plt
x = np.linspace(0,1,Nx)
y = np.linspace(0,1,Ny)
X,Y = np.meshgrid(x,y)
U = u.reshape(Ny,Nx)
clev = np.linspace(0,U.max(),100)
plt.contourf(X,Y,U, clev)
plt.colorbar()
plt.show()
