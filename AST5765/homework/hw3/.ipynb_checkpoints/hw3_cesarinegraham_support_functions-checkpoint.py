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
    x : Input as int, float, complex, or numpy.ndarray

    Returns
    -------
    int, float, complex, or numpy.ndarray - the square of the input.

    Notes
    -----
    math:: square(x) = x^(2) = x * x

    Examples
    --------
    >>> square(8) #int
    8
    
    >>> a = np.array([4, 2, 3]) # 1D array
    >>> square(a)
    array([16,  4,  9])
    
    >>> d = np.array([ [4, 2, 3], [2, 9, 7], [1, 2, 3] ]) #3D array
    >>> square(d)
    array([[16,  4,  9],
           [ 4, 81, 49],
           [ 1,  4,  9]])
           
    >>> square(3.7) #float
    13.690000000000001

    >>> square(3 + 4j) #complex
    (-7+24j)
           
    Revisions
    ---------
    2025-09-05 ce970069@ucf.edu downloaded the "module_template_with_docstring.py"
        from Webcourses, renamed it "hw3_cesaringraham_support_function.py".
    2025-09-06 ce970069@ucf.edu removed some comments and optimized the file for my own
        function- essentially 'cleaned' the file
    2025-09-06 ce970069@ucf.edu tested several examples including int 8, 1d array [4,2,3], 3d
        array [[4, 2, 3], [2, 9, 7], [1, 2, 3]], float 3.7, and complex square(3 + 4j). Added 
        TypeError to raise when an input is not an int, float, complex, or np.ndarray. Verified 
        the error for a string input.
    2025-09-06 ce970069@ucf.edu added more comments, deleted unnecessary libraries.
    
    """
#main body of the function
    
    #raising errors if odd-inputs    
    if type(x) not in (int, float, complex, np.ndarray):
        raise TypeError("Input must be int, float, complex, or numpy.ndarray!")

    #actual math of the function
    square = x * x
        
    return square



#squareplot function
def squareplot(low, high, points, saveplot = False) :
    #this is our function's doc, string!
    """ This function is to plot the squares of numbers over a specified range.
    
    Parameters
    ----------
    low : float
        The starting value of the range.
    high : float
        The inclusive, ending value of the range.
    points : int
        The number of points to generate in the range.
    saveplot : str or bool, optional
        If a string (filename) is provided, the plot is saved as a PDF with that name.
        If False (default), the plot is only displayed and not saved.

    Returns
    -------
    Plot

    Examples
    --------
    >>> squareplot(0, 5, 6)  
    plot #upward facing curve

    >>squareplot(0, 5, 6, saveplot="hw3_cesarinegraham_squareplot.pdf")  
    plot, .pdf of plot
           
    Revisions
    ---------
    2025-09-05 ce970069@ucf.edu created squareplot
    
    """
#main body of the function
    
  #creating array of evenly spaced points including the high end
    x = np.linspace(low, high, points)
    
    # calling square to get y
    y = square(x)
    
    # plotting x vs y
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Square Function")
    plt.grid(True)
    
    # saving plot (optional)
    if saveplot:
        plt.savefig(saveplot, format='pdf') #saves the plot as a pdf if declared to do so
    
    plt.show()
    
