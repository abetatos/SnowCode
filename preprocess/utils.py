import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import fiona
from os import listdir
from os.path import isfile, join
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import resize
import os.path
from shapely.geometry import shape, Polygon
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import geojson
import subprocess
from osgeo import gdal
from datetime import date
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from datetime import date, timedelta
from scipy import stats

def show(raster, title=None):
    plt.imshow(raster, cmap='inferno', interpolation='none')
    plt.title(title)
    plt.colorbar()
    plt.show()


def access_raster(path, aoi=None):

    if aoi == None:
        with rasterio.open(path) as src:
            array = src.read()
            meta = src.meta
            transform = src.meta['transform']
            extent = src.bounds
            extent_dims = {'north': extent.top, 'south': extent.bottom, 'west': extent.left, 'east': extent.right}
            polygon_extent = polygon_generator(extent_dims)

        return {'array': array, 'meta': meta, 'transform': transform, 'extent': extent, 'polygom': polygon_extent}

    else:
        with fiona.open(aoi, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        
        with rasterio.open(path) as src:
            array, transform = rasterio.mask.mask(src, shapes, nodata=0, crop=True)
            meta = src.meta
            extent = src.bounds
        return {'array': array, 'meta': meta, 'transform': transform, 'extent': extent}


def write_single_channel_gtiff(raster, transform, meta,  out_path, nodata=np.nan):
    assert len(raster.shape) == 2
#     print(f'Writing...{out_path}')
    with rasterio.open(str(out_path),
                           mode='w',
                           crs=meta['crs'],
                           driver=meta['driver'],
                           nodata=nodata,
                           dtype=raster.dtype,
                           count=meta['count'],
                           height=raster.shape[0],
                           width=raster.shape[1],
                           transform=transform
                           ) as dst:
        dst.write(raster, 1)


def get_paths_in_folder(folder):
    products = [f for f in listdir(folder) if isfile(join(folder, f))]
    full_paths = []
    for product in products:
        full_paths.append(fr'{folder}/{product}')
    return full_paths


def generate_slope_filter(aoi=None):
    slope = r'D:\Snow_Depth_Data\ancillary\eu_alps\DEM-slope\slope_wgs84.tif'

    if aoi == None:
        aoi = r'D:\Snow_Depth_Data\timeseries_testdata\20-12-07_20-12-09\training_data\aoi_shape\cropper_shape.shp'
        slope_full = access_raster(slope, aoi)
        slope = slope_full['array'][0]
        slope[slope > 50] = 50
        slope[slope < 50] = 1
        slope[slope == 50] = 0
        return slope

    else:
        slope_full = access_raster(slope, aoi)
        slope = slope_full['array'][0]
        slope[slope > 50] = 50
        slope[slope < 50] = 1
        slope[slope == 50] = 0
        return slope


def generate_aspect_filters(aspect_path, angle=180, aoi=None):

    aspect = aspect_path

    aspect_full = access_raster(aspect, aoi)
    ascending = aspect_full['array'][0]
    ascending[ascending < angle] = 0
    ascending[ascending != 0] = 1

    aspect_full = access_raster(aspect, aoi)
    descending = aspect_full['array'][0]
    descending[descending < angle] = 0
    descending[descending != 0] = 1
    descending[descending == 0] = 2
    descending[descending == 1] = 0
    descending[descending == 2] = 1

    return {'ascending': ascending,
            'descending': descending,
            'transform': aspect_full['transform'],
            'meta': aspect_full['meta']
            }



def get_sentinel_product_info(path_to_product):

    product_name = path_to_product[-71:]
    extension = os.path.splitext(product_name)[1]
    filename = os.path.splitext(product_name)[0]

    mission = filename[0:3]
    mode = filename[4:6]
    product_type = filename[7:10]
    resolution_class = filename[10:11]
    processing_level = filename[12:13]
    polarisation = filename[14:16]
    date = filename[17:25]
    start_time = filename[17:32]
    stop_time = filename[33:48]
    absolute_orbit = filename[49:55]
    mission_data_take_id = filename[56:62]
    product_unique_id = filename[63:67]

    return {
    'mission': mission,
    'mode': mode,
    'date': date,
    'Product type':product_type,
    'processing_level':processing_level,
    'resolution class':resolution_class,
    'polarisation':polarisation,
    'start time':start_time,
    'stop time': stop_time,
    'absolute orbit': absolute_orbit,
    'mission data take id': mission_data_take_id,
    'product unique id':product_unique_id,
    'extension': extension
    }


def get_shapefile_data(sf):
    collection = fiona.open(sf)
    country = next(iter(collection))
    extent = shape(country['geometry']).bounds
    n = round(extent[1], 6)
    s = round(extent[3], 6)
    e = round(extent[2], 6)
    w = round(extent[0], 6)
    return {
        'north': n,
        'south': s,
        'east': e,
        'west': w
    }

def shapefile_to_pgstring(sf, buffer):

    ex = get_shapefile_data(sf)

    return Polygon([[ex['west'], ex['south']],
                    [ex['west'], ex['north']],
                    [ex['east'], ex['north']],
                    [ex['east'], ex['south']],
                    [ex['west'], ex['south']]]).buffer(buffer, cap_style=3)


def polygon_generator(ex):
    return Polygon([[ex['west'], ex['south']],
                    [ex['west'], ex['north']],
                    [ex['east'], ex['north']],
                    [ex['west'], ex['south']]])


def is_product_ascending_or_descending(product_name):

    products = api.query(identifier=product_name)
    product_id = list(products)[0]
    product_data = api.get_product_odata(product_id, full=True)
    # print(product_data)
    return str((product_data['Pass direction']))

def calculate_slope(DEM):
    gdal.DEMProcessing('slope.tif', DEM, 'slope')
    with rasterio.open('slope.tif') as dataset:
        slope=dataset.read(1)
    return slope

def calculate_aspect(DEM):
    gdal.DEMProcessing('aspect.tif', DEM, 'aspect')
    with rasterio.open('aspect.tif') as dataset:
        aspect=dataset.read(1)
    return aspect


def ASF_json_downloader(json_path):

    with open(json_path) as f:
        gj = geojson.load(f)

    download_urls = []
    for index in range(len(gj['features'])):
        download_urls.append(gj['features'][index]['properties']['url'])

    print(len(download_urls))

    for url in download_urls[15:19]:

        chromesearch = 'start chrome'

        subprocess.check_output(f'{chromesearch} {url}', shell=True)

def interval_checker(data_dir):

    products = get_paths_in_folder(data_dir)
    for index in range(len(products)):
        try:
            d0 = date(year=int(products[index][-54:-50]),
                      month=int(products[index][-50:-48]),
                      day=int(products[index][-48:-46]))

            d1 = date(year=int(products[index + 1][-54:-50]),
                      month=int(products[index + 1][-50:-48]),
                      day=int(products[index + 1][-48:-46]))

            delta = d1 - d0

            if str(delta)[0:2] != '12':
                print(f'WARNING: There is a data gap between {products[index]} and {products[index+1]}')
        except:
            pass

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def lin_regression_analysis(x, y):
    x = np.array(x)
    y = np.array(y)
    x = x.flatten()
    y = y.flatten()
    mask = ~np.isnan(x) & ~np.isnan(y)
    n = mask.shape[0]
    line_slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
    rmse = np.sqrt(mean_squared_error(x[mask],y[mask]))
    mae = mean_absolute_error(x[mask], y[mask])
    return {'m':line_slope, 'b':intercept, 
            'r': round(r_value, 2), 'r2':round(r_value**2, 2), 
            'p':round(p_value, 3), 'n':n,
            'rmse':rmse, 'mae':mae, 
            'mask':mask}

def linear_function(x, m, b):
    return m * x + b

def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c

def poly_reg(x, y, show_it=False):
    nan_array = np.isnan(x)
    mask = ~ nan_array
    X = x[mask].flatten()
    y = y[mask].flatten()
    v = curve_fit(quadratic_function, X, y)[0]
    y_pred = quadratic_function(X, v[0], v[1], v[2])
    if show_it:
        plt.scatter(X, y, s=0.0001)
        plt.scatter(X, y_pred, s=1)
        plt.show()
    return [v[0], v[1], v[2]]

class ProductChecker:
    
    def __init__(self, dates):
        self.dates = dates

    def get_date_difference(self, y_1, m_1, d_1, y_2, m_2, d_2):
        d0 = date(y_1, m_1, d_1)
        d1 = date(y_2, m_2, d_2)
        delta = d1 - d0
        return delta.days

    def get_timeseries_differences(self):
        time_differences = []
        for index in range(len(d_dates)):
            try:
                y_1, m_1, d_1 =  int(self.dates[index][0:4]), int(self.dates[index][4:6]), int(self.dates[index][6:8])
                y_2, m_2, d_2 =  int(self.dates[index + 1][0:4]), int(self.dates[index + 1][4:6]), int(self.dates[index + 1][6:8])
                time_differences.append(self.get_date_difference(y_1, m_1, d_1, y_2, m_2, d_2))
            except:
                pass
        print(time_differences)




def sig_to_depth(signal):
    '''PolSAR algorithm calibrated to SLF point data over static regions'''
    depth = 54.953 * signal + 0.4278
    return depth

def moving_avg(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    ans = (cumsum[n:] - cumsum[:-n]) / float(n)
    ans = list(ans)
    ans.insert(0, x[0])
    ans.insert(1, x[1])
    return ans


import boto3
s3 = boto3.resource('s3')

def download_sce_from_s3(bucket_name, s3_folder, local_dir=None):
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        date = obj.key[-25:-17]
        # Check the map has not already been downloaded to the VM 
        if os.path.isfile(fr'{local_dir}/{obj.key}'):
            pass
        # Otherwise download it to the VM
        else:      
            target = obj.key if local_dir is None \
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == '/':
                continue
            bucket.download_file(obj.key, target)
            print(fr'{local_dir}/{obj.key} loaded into VM')



def create_date_range(start, finish, form='%Y%m%d'):
    start_date = date(start[0], start[1], start[2]) 
    end_date = date(finish[0], finish[1], finish[2])
    delta = end_date - start_date  # as timedelta
    date_series = []
    # include the start and end day, if want to remove these 2 day, use range(1, delta.days) 
    for i in range(delta.days + 1):  
        day = start_date + timedelta(days=i)
        date_series.append(day.strftime(form))
    return date_series

def res_resampler(gtiff_in, gtiff_out, xres, yres):
    tiff_in = gdal.Open(gtiff_in)
    resampled = gdal.Warp(
        gtiff_out,
        tiff_in, 
        xRes=xres, 
        yRes=yres, 
        resampleAlg="bilinear"
    )