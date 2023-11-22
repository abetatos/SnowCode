from osgeo import gdal
import numpy as np
parent_path = '/home/ubuntu'

def view (offset_y, offset_x, shape, step=1):
    """
    Function returning two matching numpy views for moving window routines.
    - 'offset_y' and 'offset_x' refer to the shift in relation to the analysed (central) cell
    - 'shape' are 2 dimensions of the data matrix (not of the window!)
    - 'view_in' is the shifted view and 'view_out' is the position of central cells
    (see on LandscapeArchaeology.org/2018/numpy-loops/)
    """
    size_y, size_x = shape
    x, y = abs(offset_x), abs(offset_y)
   
    x_in = slice(x , size_x, step)
    x_out = slice(0, size_x - x, step)

    y_in = slice(y, size_y, step)
    y_out = slice(0, size_y - y, step)
 
    # the swapping trick    
    if offset_x < 0: x_in, x_out = x_out, x_in                                
    if offset_y < 0: y_in, y_out = y_out, y_in
 
    # return window view (in) and main view (out)
    return np.s_[y_in, x_in], np.s_[y_out, x_out]

def tpi(elevation_model, r, output_model):
    """ Non disclosed function
    """