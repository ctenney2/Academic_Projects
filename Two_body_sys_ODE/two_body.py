#!/bin/anaconda3/bin/python

import math
import sympy as sym
import math
import numpy as np
from numpy import linalg as la
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("Lets do the 2-body problem")

m_1 = 1.0
m_2 = 1.0

G = 6.67408e-11 #m^3 kg^-1 s^-2
print("G is ",G+1)
x_1 = np.array([2.0, 2.0, 2.0])

x_2 = np.array([5.0, 5.0, 5.0])

v_1 = np.array([5.0, 0.0, 0.0])

v_2 = np.array([0.0, 0.0, 0.0])


print("norm is ", la.norm(x_2))

F_12 = G*m_1*m_2

F1 = 1*(x_2-x_1)/(la.norm(x_2-x_1)**3)
print("F1 is ",F1)

#Lets make a function F for the force that takes in two vectors, and two masses
def F(x1,x2,m1,m2):
    #G = 6.67408e-11 #m^3 kg^-1 s^-2 Outside world does not need this constant
    G = 1.0 #For testing

    return G*m1*m2*(x1-x2)/(la.norm(x1-x2)**3)

#End function

print("function f is",F(x_1,x_2,m_1,m_2))
print("x component is ",F(x_1,x_2,m_1,m_2)[0])
#This works

ddx_1 = -F(x_1,x_2,m_1,m_2)/m_1
ddx_2 = -F(x_2,x_1,m_2,m_1)/m_2


#Lets make a vector y that will give is a 1st order ode

#y = np.array([x_1,dx_1,x_2,dx_2])
#then we will have 4 equations:
# 
# We have the 4 system of equations
# dy1 = y2
# dy2 = -(y1 - y3)/(norm(y1-y3)**3)
# dy3 = y4
# dy4 = -(y3 - y1)/(norm(y3-y1)**3)



def func(y, t):
    dy = [y[1],
         -(y[0]-y[2])/(la.norm(y[0]-y[2])**3),
          y[3], 
         -(y[2]-y[0])/(la.norm(y[2]-y[0])**3)]

    return dy
#end func
yx = np.array([x_1[0],v_1[0],x_2[0],v_2[0]])
yy = np.array([x_1[1],v_1[1],x_2[1],v_2[1]])
yz = np.array([x_1[2],v_1[2],x_2[2],v_2[2]])
t = np.arange(0,6.0,0.001)
#print("funct", func(yx,t))

solx = integrate.odeint(func,yx,t)
soly = integrate.odeint(func,yy,t)
solz = integrate.odeint(func,yz,t)
#print("Solution is",soly[:,2])
plt.plot(t,solz[:,0])
plt.plot(t,solz[:,2])
plt.show()

solr1 = (solx[:,0]**2 + soly[:,0]**2 + solz[:,0]**2)**(1.0/2.0)
solr2 = (solx[:,2]**2 + soly[:,2]**2 + solz[:,2]**2)**(1.0/2.0)
plt.plot(t,solr1)
plt.plot(t,solr2)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(solx[:,0], soly[:,0], solz[:,0], c='r', marker='o')
ax.scatter(solx[:,2], soly[:,2], solz[:,2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()





