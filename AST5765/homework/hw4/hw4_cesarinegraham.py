#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cesarine Graham
# AST 5765 Homework 4
# 09/09/2025


# In[2]:


#defining my libraries
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Problem 2a. 
# Find a python fun. that draws a random sample of N draws from a Gaussian distribution \
# of width sigma and mean x

# using this in my stat class currently!
m = 10000 
mu = 55
sigma = 13
sample = np.random.normal(mu, sigma, size=m) #Draws m i.i.d samples from N(mu, sigma2). x is an array of length m. 

print(sample)


# In[4]:


# Problem 2b
# plot the histogram of your sample
m = 10000 
mu = 55
sigma = 13
sample = np.random.normal(mu, sigma, size=m)

plt.figure( figsize = (5, 4) )
plt.hist(sample, bins=np.arange(0, 101, 1), color='green', density=False, alpha=0.6)
plt.title("Histogram of a Gaussian ($\mu=55$, $\sigma=13$, $N=10000$)")
plt.xlabel('x')
plt.ylabel('N(x)')
plt.xlim(0, 100)
plt.grid(False)
plt.savefig("prob2b_cesarinegraham_histogram.png")
plt.show()


# In[5]:


#Problem 2c
# plot the histogram with a gaussian overplot

m = 10000
mu = 55
sigma = 13
sample = np.random.normal(mu, sigma, size=m) #Draws m i.i.d samples from N(mu, sigma2). x is an array of length m. 

# bins from 0 to 100 with width 1
bins = np.arange(0, 101, 1)              # edges: 0,1,2,...,100
bin_width = bins[1] - bins[0]            # should be 1
bin_centers = bins[:-1] + bin_width/2    # centers: 0.5, 1.5, ..., 99.5

# gaussian PDF evaluated at bin centers (continuous PDF)
pdf = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((bin_centers - mu)/sigma)**2)

# approximate counts in each bin by pdf(center) * N * bin_width
approx_counts = pdf * m * bin_width

# plotting histogram and pdf
plt.hist(sample, bins=bins, density=False, alpha=0.6, color='green', label='Sample counts')
plt.plot(bin_centers, approx_counts, color='red', linewidth=2, marker='o', markersize=4, label='Gaussian approx')

# formatting the plot
plt.xlabel('x')
plt.ylabel('N(x)')
plt.title("Histogram of a Gaussian ($\mu=55$, $\sigma=13$, $N=10000$)")
plt.xlim(0, 100)
plt.grid(False)
plt.savefig("prob2c_cesarinegraham_histogram2.png", dpi=300)
plt.show()


# #### Problem 3a.
# 
# Sources used: 
# - (Belvington 2003)
# - https://www.cmrr.umn.edu/stimulate/frame/fwhm/node1.html
# 
# We will start from the Gaussian function:
# 
# $$
# f(x) = A\,\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big),
# $$
# 
# where $A$ is the peak amplitude (at $x=\mu$), $\mu$ is the mean, and $\sigma$ is the standard deviation.
# 
# The maximum value is at $x=\mu$:
# 
# $$
# f(\mu)=A.
# $$
# 
# The FWHM is the distance between the two $x$-values where the function equals half the peak:
# 
# $$
# f(x) = \frac{A}{2}.
# $$
# 
# We will set the Gaussian equal to half the peak:
# 
# $$
# A\,\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big)=\frac{A}{2}.
# $$
# 
# Divide both sides by $A$ and take natural log:
# 
# $$
# \exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big)=\frac{1}{2}
# $$
# $$
# -\frac{(x-\mu)^2}{2\sigma^2}=\ln\!\left(\tfrac{1}{2}\right)=-\ln 2.
# $$
# 
# Solve for $|x-\mu|$:
# 
# $$
# \frac{(x-\mu)^2}{2\sigma^2}=\ln 2
# $$
# $$
# |x-\mu|=\sigma\sqrt{2\ln 2}.
# $$
# 
# Recall that the FWHM is the distance between the two symmetric solutions, so $x=\mu\pm\sigma\sqrt{2\ln2}$:
# 
# $$
# \text{FWHM} = 2\sigma\sqrt{2\ln 2}.
# $$
# $$
# \sqrt{2\ln2}\approx\sqrt{2\times 0.693147\ldots}=\sqrt{1.386294\ldots}\approx 1.177410\ldots
# $$
# $$
# \text{FWHM}\approx 2\times 1.177410\;\sigma \approx 2.354820\;\sigma.
# $$
# $$
# \boxed{\therefore \text{FWHM} = 2\sqrt{2\ln 2}\;\sigma \approx 2.35482\,\sigma,}
# $$
# 

# #### Problem 3b.
# 
# Sources used: 
# - (Belvington, 2003)
# - https://math.stackexchange.com/questions/3245738/finding-the-equation-of-a-straight-line-on-a-log-log-plot-given-two-points
# 
# Given a straight line in loglog space:
# $$ \log(y) = m \cdot \log(x) + b $$
# 
# Taking the exponential of both sides we get:
# $$
# \begin{align}
# e^{\log(y)} &= e^{m \cdot \log(x) + b} \\
# &= e^{m \cdot \log(x)} \cdot e^b \\
# &= \left(e^{\log(x)}\right)^m \cdot e^b
# \end{align}
# $$
# 
# which is,
# $$ y = x^m \cdot e^b $$

# In[ ]:




