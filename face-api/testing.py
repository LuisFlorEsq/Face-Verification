import numpy as np

def generate_array(shape, fill_type='ones'):
    """
    Generates an n-dimensional NumPy array.

    Parameters:
        shape (tuple): Shape of the array (e.g., (2, 3), (3, 4, 5)).
        fill_type (str): 'ones' to fill with ones, 'random' to fill with random values between 0 and 1.

    Returns:
        numpy.ndarray: Generated array.
    """
    if fill_type == 'ones':
        return np.ones(shape)
    elif fill_type == 'random':
        return np.random.rand(*shape)
    else:
        raise ValueError("fill_type must be either 'ones' or 'random'")
    
    
array1 = generate_array((1, 128), fill_type='ones')
print(array1.tolist())