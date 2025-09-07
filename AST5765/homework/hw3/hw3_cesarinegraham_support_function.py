"""This support function will calculate the square of some input and return it. 
The input can be a scalar or an array of any dimension or numerical type, 
and the function will return its square.

"""
#importing my needed libraries
import numpy as np
import matplotlib.pyplot as plt 

#square function
def square(x) :
    #this is our function's docstring!
    """This function calculates the square of some input and returns it. 
    The input can be a scalar or an array of any dimension or numerical type, 
    and the function will return its square.

    Parameters
    ----------
    x : int, float, complex, or numpy.ndarray
        Input number or array.

    Returns
    -------
    int, float, complex, or numpy.ndarray
        The square of the input.

    Notes
    -----
    math:: square(x) = x^(2) = x * x

    Examples
    --------
    >>> square(8)
    8
    
    >>> a = np.array([4, 2, 3])
    >>> square(a)
    array([16,  4,  9])
    
    >>> d = np.array([ [4, 2, 3], [2, 9, 7], [1, 2, 3] ])
    >>> square(d)
    array([[16,  4,  9],
           [ 4, 81, 49],
           [ 1,  4,  9]])
           
    >>> square(3.7)
    13.690000000000001
           
    Revisions
    ---------
    2025-09-05 ce970069@ucf.edu downloaded the "module_template_with_docstring.py"
        from Webcourses, renamed it "hw3_cesaringraham_support_function.py".
    2025-09-06 ce970069@ucf.edu removed some comments and optimized the file for my own
        function- essentially 'cleaned' the file
    
    """
    
#main body of the function
    square = x * x
    return square
    
