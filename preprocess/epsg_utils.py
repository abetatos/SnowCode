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
   
    win = np.ones((2* r +1, 2* r +1))
    r_y, r_x  = win.shape[0]//2, win.shape[1]//2
    win[r_y, r_x  ]=0  # let's remove the central cell

   
    dem = gdal.Open(elevation_model)
    mx_z = dem.ReadAsArray()

    #matrices for temporary data
    mx_temp = np.zeros(mx_z.shape)
    mx_count = np.zeros(mx_z.shape)

    # loop through window and accumulate values
    for (y,x), weight in np.ndenumerate(win):

        if weight == 0 : continue  #skip zero values !
        # determine views to extract data
        view_in, view_out = view(y - r_y, x - r_x, mx_z.shape)
        # using window weights (eg. for a Gaussian function)
        mx_temp[view_out] += mx_z[view_in]  * weight

       # track the number of neighbours
       # (this is used for weighted mean : Σ weights*val / Σ weights)
        mx_count[view_out] += weight

    # this is TPI (spot height – average neighbourhood height)
    out = mx_z - mx_temp / mx_count

    # writing output
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(output_model, mx_z.shape[1], mx_z.shape[0], 1, gdal.GDT_Float32)
    ds.SetProjection(dem.GetProjection())
    ds.SetGeoTransform(dem.GetGeoTransform())
    ds.GetRasterBand(1).WriteArray(out)
    ds = None