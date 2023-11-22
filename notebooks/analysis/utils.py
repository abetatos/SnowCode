
import rasterio
import fiona
from shapely.geometry import Polygon

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


def polygon_generator(ex):
    return Polygon([[ex['west'], ex['south']],
                    [ex['west'], ex['north']],
                    [ex['east'], ex['north']],
                    [ex['west'], ex['south']]])


