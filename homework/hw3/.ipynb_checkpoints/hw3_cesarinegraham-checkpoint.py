#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Cesarine Graham
#AST 5765 Homework 3
#09/05/2025


# In[2]:


#importing my libraries
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from hw3_cesarinegraham_support_functions import square


# In[4]:


# example for the support function docstring
square (8)


# In[5]:


# example for the support function docstring
a = np.array([4, 2, 3])
square(a)


# In[6]:


# example for the support function docstring
d = np.array([
    [4, 2, 3], 
    [2, 9, 7],
    [1, 2, 3]
])
square(d)


# In[7]:


# example for the support function docstring
square(3.7)


# In[8]:


# example for the support function docstring
square(3 + 4j)


# In[9]:


# example for the support function docstring
# square("word") #making sure the typeerror works
#commented this out so i can continue running without an error


# In[10]:


#problem 2h
#create an array test_square_1 with integers from 0 to 9. Import and call function square to square array test_square_1 and print the result

test_square_1 = np.arange(10) #array with int from 0 to 10
print(test_square_1) #checking the array

from hw3_cesarinegraham_support_functions import square #importing square
square(test_square_1) 


# In[11]:


# problem 2i
# In a single line, create a 5x5 array test_square_2 that contains floats from 0 to 25. 
# Call function square to square array test_square_2 and print the result.

test_square_2 = np.linspace(0, 25, 25).reshape(5, 5) #5x5 array w floats from 0 to 25
print(test_square_2) #checking the array

square(test_square_2) 


# In[12]:


#problem 3
from hw3_cesarinegraham_support_functions import squareplot #importing squareplot

# example for the docstring
squareplot(0, 5, 6)  

#example for the docstring
# squareplot(0, 5, 6, saveplot="hw3_cesarinegraham_squareplot_example.pdf")  


# In[13]:


# problem 3
# call squareplot to plot the squares of the numbers 1, 2.5, 4, 5.5, and 7, and save to an appropriate named file

from hw3_cesarinegraham_support_functions import squareplot #importing squareplot

squareplot(1, 7, 5, saveplot="hw3_cesarinegraham_squareplot.pdf")  #calls 5 evenly spaced points from 1 to 7


# In[ ]:




