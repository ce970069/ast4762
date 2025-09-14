#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits


# In[7]:


# making 300x200 array
nx = 300
ny = 200


# #### With a Loop: ####

# In[18]:


# with a loop
wanted_array = np.zeros((nx, ny) , dtype = np.float64)
for i in range ( nx ):
    wanted_array[i,:] = i

# plotting
plt.imshow (wanted_array, cmap = 'gray')
plt.gca().invert_yaxis()
dm = plt.show()

# different way to plot
plt.contourf(wanted_array, cmap = 'gray')
dm = plt.show()


# #### Without a loop: ####

# In[33]:


# my attempt
x_array = np.arange(0,200,1)
y_array = np.arange(0,300,1)
np.repeat

#solution 1 with the repeat:
init = np.arange(nx, dtype=np.float64).reshape(nx,1)
wanted_array_v2 = np.repeat(init, ny, axis=1)

plt.imshow (wanted_array_v2, cmap='gray')
dm = plt.gca().invert_yaxis()


# In[34]:


# solution 2 broadcasting

wanted_array_v3 = np.zeros( (nx,ny), dtype=np.float64)
init2 = np.arange( nx ).reshape ( nx,1 )
wanted_array_v3[:,:] = init2
plt.imshow (wanted_array_v3, cmap='gray')
dm = plt.gca().invert_yaxis()


# In[ ]:


#solution 3 , brute force plus reshape

wanted_array_v4 = np.arange (nx*ny, dtype=np.float64)
wanted_array_v4 /= ny


# #### Now with AI: ####

# In[35]:


import numpy as np
import matplotlib.pyplot as plt
import random

# dimensions
width, height = 300, 200

# --- Method 1: Using loops ---
arr_loop = np.zeros((height, width), dtype=np.float64)
for y in range(height):
    for x in range(width):
        arr_loop[y, x] = x   # each element gets its column index

# --- Method 2: Without loops (vectorized) ---
arr_no_loop = np.tile(np.arange(width, dtype=np.float64), (height, 1))

# --- Display the array ---
plt.imshow(arr_no_loop, cmap="gray")
plt.colorbar(label="x-coordinate value")
plt.title("Array filled with x-coordinates")
plt.show()

# --- Verify random values ---
for _ in range(5):
    y = random.randint(0, height - 1)
    x = random.randint(0, width - 1)
    print(f"arr[{y}, {x}] = {arr_no_loop[y, x]} (should be {x})")

# --- Check dtype ---
print("Data type:", arr_no_loop.dtype)


# #### FITS Data ####

# In[51]:


import astropy.io.fits as fits

im = fits.getdata('m42_40min_ir.fits') 
plt.imshow(im, cmap = 'plasma')


# In[ ]:




