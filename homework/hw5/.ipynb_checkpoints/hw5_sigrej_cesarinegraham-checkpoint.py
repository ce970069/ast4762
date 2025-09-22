""" sigrej() performs iterative sigma rejection ("sigma clipping")
on an input dataset. Outliers further than Nσ from the mean
are flagged as bad (False) in the mask. Multiple iterations
can be run by providing a tuple of sigma limits (e.g. (5., 5.)).
The final Boolean mask indicates which data points survived.

"""

#importing my needed libraries
import numpy as np

def sigrej(data, limits=(5., 5.), mask=None):
    """
    Sigma rejection routine.
    
    Parameters
    ----------
    data : ndarray
        Input data array.
    limits : tuple of floats
        Rejection limits (number of sigmas for each iteration).
    mask : ndarray, optional
        Boolean mask of same shape as data (True = good, False = bad).
        If None, all data is initially considered good.
    
    Returns
    -------
    mask : ndarray
        Final Boolean mask after sigma rejection.

    Examples
    --------
    >>> N = 10000                                                       # number of photons
    >>> test_data_ccd = np.zeros(400)                                   # creating initial array
    >>> test_data_ccd[:395] = np.random.poisson(N, 395)                 # adding values to the array
    >>> test_data_ccd[395:] = np.random.uniform(0, 1e6, 5)              # adding noise to the array
    >>> print("Raw data mean:", np.mean(test_data_ccd))
    >>> print("Raw data median:", np.median(test_data_ccd))
    >>> final_mask_test = sigrej(test_data_ccd, limits=(5., 5.))    
    >>> filtered_data = test_data_ccd[final_mask_test]         
    >>> print("\nFiltered data mean:", np.mean(filtered_data))
    >>> print("Filtered data median:", np.median(filtered_data))
    Raw data mean: 17673.847826367477
    Raw data median: 9996.5
    Filtered data mean: 9997.774683544303
    Filtered data median: 9995.0
    
    >>> gauss_data = np.random.normal(loc=50, scale=3, size=500)        # creating initial array with gaussian dist.
    >>> gauss_data[::100] = [200, -100, 300, -250, 500]                 # adding some outlier data points
    >>> print("Raw mean:", np.mean(gauss_data))
    >>> print("Raw median:", np.median(gauss_data))
    >>> mask = sigrej(gauss_data, limits=(3., 3.))                      # applying the mask
    >>> filtered = gauss_data[mask]
    >>> print("\nFiltered mean:", np.mean(filtered))
    >>> print("Filtered median:", np.median(filtered))
    Raw mean: 50.46261550829812
    Raw median: 49.755351602852954
    Filtered mean: 49.63879975413571
    Filtered median: 49.731646128464305

    >>> data_small = np.array([10, 11, 9, 10, 12, 11, 10, 9, 13, 200])  # creating small array by hand
    >>> print("Raw mean:", np.mean(data_small))
    >>> print("Raw median:", np.median(data_small))
    >>> mask = sigrej(data_small, limits=(3., 3.))                      # applying the mask
    >>> filtered = data_small[mask]
    >>> print("Filtered mean:", np.mean(filtered))
    >>> print("Filtered median:", np.median(filtered))
    Raw mean: 29.5
    Raw median: 10.5
    Filtered mean: 29.5
    Filtered median: 10.5

    Revisions
    ---------
    2025-09-21 ce970069@ucf.edu created the function.
    2025-09-21 ce970069@ucf.edu tested the function and added the 
        examples portion of the docstring. Tested two arrays using
        np.random.poison + np.random.uniform, and np.random.normal.
        Third example is an array made by hand.
    """
# main body of the function

    #initialize if mask is not provided
    if mask is None:
        mask = np.ones(data.shape, dtype=bool)
    
    for nsig in limits:
        good_data = data[mask]
        med = np.mean(good_data)
        sigma = np.std(good_data)

        low_range = med - nsig * sigma
        high_range = med + nsig * sigma

        mask = mask & (data > low_range) & (data < high_range)
    
    return mask
