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



#System of ode solver
#Uses the Runge-Kutta Method for Systems of Differential Equations: Chapter 5.9 Algorithm 5.7
def ode_solve(f, a, b, m, N, ic):

    h = (b-a)/N
    t = a

    #Store the output in a 2D array. 1st component is time, 2nd component is the input parameters
    sol = np.zeros([N+1,len(ic)])
                    #To hold the initial contiditons + the ammount of iterations
    #Save initial contiation to t = a
    sol[0,:] = ic

    w = ic
    i = 1
    while(i <= N):
        print("t and i ",t, i)
        k1 = h*f(w,t)
        print("k1 is",k1)
        k2 = h*f(w + 0.5*k1, t + h/2.0)
        k3 = h*f(w + 0.5*k2, t + h/2.0)
        k4 = h*f(w + k3, t + h)

        w = w + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        
        t = a + i*h
        
        
        sol[i,:] = w
        i = i + 1

        
    #End While
    return sol



#Edn ode_solve

print("array test")
N = 10
ic = [0,1,2,3]
w = np.zeros([N,len(ic)])
w[0,:] = ic
#print ("w is ",w)

#Test
#Y = [u1,u2,u3]
def ft(Y,t):
   dy = np.array([ Y[1], -Y[0] - 2.0*math.exp(t) + 1.0, -Y[0] - math.exp(t) + 1.0])

   return dy
#end
y = np.array([1.0,0.0,1.0])
N=4
sol =  ode_solve(ft, 0.0, 2.0,3, N, y)

print("sol is ",sol)
#[[ 1.          0.          1.        ]
# [ 0.70787076 -1.24988663  0.39884862]
# [-0.33691753 -3.01764179 -0.29932294]
# [-2.41332734 -5.40523279 -0.92346873]
# [-5.89479008 -8.70970537 -1.32051165]]

#And we get the books result




