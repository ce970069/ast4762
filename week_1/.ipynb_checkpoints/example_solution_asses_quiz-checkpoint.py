#!/usr/bin/env python
# coding: utf-8

# ### Wallis Product of $\pi$:
# 
# $\frac{pi}{2} = \Pi_{n=1}^{n=\infty} \frac{4n^2}{4n^2-1}$

# In[ ]:


import numpy as np

def wallis_pi(N):
    pi_wal = 1
    
    if N <0:
        raise Exception('cannot have negative numbers! Please choose another N')
        
    if type(N) != int:
        print('You gave a non-integer value. Will use', int(N), ' instead')
        N = int(N)
    
    for n in range(1, N):
        pi_wal = pi_wal * (4 *n**2)/ (4*n**2 -1 )
        
    
    return 2* pi_wal
    


# In[ ]:


print( wallis_pi(-3))


# In[ ]:


print( wallis_pi(10000))


# In[ ]:


print( wallis_pi(10000.4))

