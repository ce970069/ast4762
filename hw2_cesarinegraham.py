# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 17:25:05 2025

@author: cesay
"""

# Cesarine Graham
# AST 5765 Homework 2
# 08/26/2025

#importing some useful packages
import numpy as np
import matplotlib.pyplot as plt

#problem 2
x = np.arange(0,1001) #ending at 1001 to include 1000
print(f'There are {len(x)} elements needed.') #printing the number of elements
print(f'The data type is:  {type(x)}' ) #printing the datatype

#printing the max and min
print ( f'The array minimum is {np.min(x)} and the array maximum is {np.max(x)}' )

#rescaling x to contain values 0 to 2pi
print(2*np.pi)
x = ( (x - 0) / (1000 - 0)) * 2*np.pi #creating a normalization factor to multiply to the array with the mins and maxs
print(x) #making sure the values are correct, should be 0 to 2pi

#printing the max and min
print ( f'The array minimum is {np.min(x)} and the array maximum is {np.max(x)}' )

#make an array y whose values are the sine values of x
y = np.sin(x)
print(y) #making sure the values are correct

#print the value of element 234
print( y[234] )


#problem 3
#plotting y vs x (publication ready) 
plt.figure( figsize = (10, 8) )
plt.plot( x, y, color = 'green' , linewidth = 3 )
plt.xlabel( 'x', fontsize = 14 )
plt.ylabel( 'sin(x)', fontsize = 14 )
plt.title('Sine Function', fontsize=16)

#saving the graph as png
plt.savefig('hw2_cesarinegraham_prob3_graph1.png')


#problem 4
#making a ramp array r with 101 evenly spaced elements going from -1 to 1
#will use linspace instead of arange so i can specify the number of elements
r = np.linspace(-1, 1, 101).astype(float) #found on NumPy documentation
print(r)

#using np.where
clipped_r = np.where( r > 0.5, 0.5, r )                     #any value greater than 0.5 is set to 0.5
clipped_r = np.where( clipped_r < -0.5, -0.5, clipped_r )   #any value less than -0.5 is set to -0.5

print( clipped_r ) #making sure everyting is alright!

#plotting r and clipped_r (publication ready) 
x = np.arange(0,101,1) #making the x-axis from 0 to 100

plt.figure( figsize = (10, 8) )
plt.plot( x, r , color = 'blue', linewidth = 3, label='r')                      #x vs r
plt.plot( x, clipped_r , color = 'orange', linewidth = 3 , label='clipped r')   #x vs clipped r
plt.legend()
plt.xlabel( 'x', fontsize = 14 )
plt.ylabel( 'y', fontsize = 14 )
plt.title('r and clipped r', fontsize=16)

#saving the graph as png
plt.savefig('hw2_cesarinegraham_prob4_graph2.pdf')


#problem 5
"""
Astropy:
https://www.astropy.org/
    Astropy is a open-source Python library for astronomy. It focuses on common tools needed for performing astronomy \
    and astrophysics with Python. Astropy allows you do perform tasks such as handling coordinates and times, FITS data, \
    iamge visualization, and cosmological calculations.

AMES Mars Global Climate Model Community Analysis Pipeline:
https://github.com/NASA-Planetary-Science/AmesCAP
    This astronomical software, Community Analysis Pipeline (CAP), is a set of Python 3 libraries and command-line executables. \
    The project is licensed under MIT, but cited through NASA Ames Mars Climate Modeling Center. CAP streamlines the downloading, \
    processing, and plotting of modeling outputs from the NASA Ames Mars Global Climate Model (GCM). The libraries are hosted through \
    a public-access github! There is a collection of data you can access you run with the software, as well as links to access the most \
    up-to-date data releases. I will definetly be using this resource in the future with my research in Martian geomorphology!
"""