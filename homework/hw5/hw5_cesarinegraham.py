#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cesarine Graham
# AST 5765 Homework 5
# 09/20/2025


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import astropy.io.fits as fits 


# #### 2. (10 Points) Here you will continue the sigma-clipping we worked on during Practicum3. Repeat the steps from problem 2b) on the subsample you created in 2b), to make a sub-subsample.
# 
# #### Print the new mean, median, and standard deviation. How different are the final mean and median?
# 
# #### How close is the final standard deviation to that expected for the Poisson distribution? Will this method always remove every bad pixel?

# In[3]:


# number of photons
N = 10000

# creating initial array
data_ccd = np.zeros(400)
data_ccd[:396] = np.random.poisson(N, 396)
data_ccd[396:] = np.random.uniform(0, 1e6, 4)

print("Raw data mean:", np.mean(data_ccd))
print("Raw data median:", np.median(data_ccd))

# ----- first sigma clip (subsample) -----
sigma = np.std(data_ccd)
med_ccd = np.median(data_ccd)

low_range = med_ccd - 5*sigma
high_range = med_ccd + 5*sigma

filtered_data_ccd = data_ccd[(low_range < data_ccd) & (data_ccd < high_range)]

print("\nSubsample mean:", np.mean(filtered_data_ccd))
print("Subsample median:", np.median(filtered_data_ccd))
print("Subsample std dev:", np.std(filtered_data_ccd))

# ----- second sigma clip (sub-subsample) -----
sigma_2 = np.std(filtered_data_ccd)
med_2_ccd = np.median(filtered_data_ccd)

low_range_2 = med_2_ccd - 5*sigma_2
high_range_2 = med_2_ccd + 5*sigma_2

filtered_2_data_ccd = filtered_data_ccd[(low_range_2 < filtered_data_ccd) & (filtered_data_ccd < high_range_2)]

print("\nSub-subsample mean:", np.mean(filtered_2_data_ccd))
print("Sub-subsample median:", np.median(filtered_2_data_ccd))
print("Sub-subsample std dev:", np.std(filtered_2_data_ccd))


# ##### The mean of the subsample (first sigma clip) and the sub-subsample (second sigma clip) cahnges from about 10272 to 10005. This is showing us that more outlier datapoints (bad-pixels) have been removed and the mean has shifted closer to the median, which is 10008. The medians have remained the same after each sigma clipping.
# 
# ##### The final standard deviation is about 93.5. The standard deviation of a Poisson distribution is $\sqrt{10000} = 100$. Therefore, I think our final standard deviation is pretty close!
# 
# ##### This method unfortunately will not remove every bad pixel. Some bad pixels will fall between the $5\sigma$ ranges.

# #### 3. 10 points) Make a Python routine called sigrej() that carries out the “sigma rejection” process you coded during the Practicum and above (i.e., use your code from practicum problem 2 b and 2c here, in a function).

# ##### see --> sigrej_cesarinegraham.py

# In[5]:


from hw5_sigrej_cesarinegraham import sigrej


# In[2]:


# example 1 for the sigrej function

# number of photons
N = 10000

# creating initial array
test_data_ccd = np.zeros(400)
test_data_ccd[:395] = np.random.poisson(N, 395)
test_data_ccd[395:] = np.random.uniform(0, 1e6, 5)
print("Raw data mean:", np.mean(test_data_ccd))
print("Raw data median:", np.median(test_data_ccd))

# applying the mask
final_mask_test = sigrej(test_data_ccd, limits=(5., 5.))
filtered_data = test_data_ccd[final_mask_test]
print("\nFiltered data mean:", np.mean(filtered_data))
print("Filtered data median:", np.median(filtered_data))


# In[3]:


# example 2 for the sigrej function:

# creating initial array with guassian distribution
gauss_data = np.random.normal(loc=50, scale=3, size=500)

# adding some outlier data points
gauss_data[::100] = [200, -100, 300, -250, 500]
print("Raw mean:", np.mean(gauss_data))
print("Raw median:", np.median(gauss_data))

# applying the mask
mask = sigrej(gauss_data, limits=(3., 3.))
filtered = gauss_data[mask]
print("\nFiltered mean:", np.mean(filtered))
print("Filtered median:", np.median(filtered))


# In[4]:


# example 3 for the sigrej function:

# creating initial array by hand
data_small = np.array([10, 11, 9, 10, 12, 11, 10, 9, 13, 200])
print("Raw mean:", np.mean(data_small))
print("Raw median:", np.median(data_small))

# applying the mask
mask = sigrej(data_small, limits=(5., 5.))
filtered = data_small[mask]
print("Filtered mean:", np.mean(filtered))
print("Filtered median:", np.median(filtered))


# In[ ]:




