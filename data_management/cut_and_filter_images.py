"""
INSTITUTO NACIONAL DE PESQUISAS ESPACIAIS
COMPUTACAO APLICADA

CODE: DATA MANAGER
AUTHOR: MATEUS DE SOUZA MIRANDA, 2022

"""

# -------- LIBRARY
# Directory
import os
import glob
import shutil

# Geospatial
from osgeo import gdal

# Data
import numpy as np

# -------- CROP THE IMAGE

# Variable
clip_count = 0

# Loop for clipping
for raster in glob.iglob('../data/render/*.tif'):
    file = gdal.Open(raster)
    file_gt = file.GetGeoTransform()

    print(raster)
    # Get coordinates of upper left corner
    xmin = file_gt[0]
    ymax = file_gt[3]
    res = file_gt[1]

    # Determine total length of raster
    xlen = res * file.RasterXSize
    ylen = res * file.RasterYSize

    # Number of tiles in x and y direction
    height = file.RasterXSize
    width = file.RasterYSize

    # Obtaining the exact value for measurement of the patches
    xdiv = int(height / 256)
    ydiv = int(width / 256)

    print('X image ', file.RasterXSize)
    print('Y image ', file.RasterYSize)
    print('X clip ', file.RasterXSize / xdiv)
    print('Y clip ', file.RasterYSize / ydiv)
    
    # size of a single tile
    xsize = xlen / xdiv
    ysize = ylen / ydiv

    # create lists of x and y coordinates
    xsteps = [xmin + xsize * i for i in range(xdiv + 1)]
    ysteps = [ymax - ysize * i for i in range(ydiv + 1)]

    # Creating directories
    name_file = os.path.basename(raster)
    if os.path.isdir(name_file) == False:
        os.mkdir('images/clip/' + name_file + '_clipped')

    # loop over min and max x and y coordinates
    for i in range(xdiv):
        for j in range(ydiv):
            xmin = xsteps[i]
            xmax = xsteps[i + 1]
            ymax = ysteps[j]
            ymin = ysteps[j + 1]

            # use gdal warp
            gdal.Warp('images/clip/' + name_file + '_clipped' + '/' + name_file+'_' + str(i) + str(j) + ".tif",
                      file, outputBounds=(xmin, ymin, xmax, ymax), dstNodata=None)

            # or gdal translate to subset the input raster
            gdal.Translate('images/clip/' + name_file + '_clipped' + '/' + name_file+'_' + str(i) + str(j) + ".tif",
                           file, projWin=(xmin, ymax, xmax, ymin), xRes=res, yRes=-res)

    # close the open dataset!!!
    dem = None
    clip_count = clip_count + 1
    #print('- Raster ', clip_count, 'Ok!')


# -------- REMOVING NON-DATA IMAGE

# Variables
nodata = 0
hasdata = 0

# Looping
for raster_clipped in glob.iglob('images/clip/**/*.tif'):
    file = gdal.Open(raster_clipped)
    ia = file.ReadAsArray()

    # Get the full path of the images
    path = os.path.dirname(raster_clipped)
    file_name = os.path.basename(raster_clipped)
    full_address = path + '/' + file_name

    m = np.nanmean(ia)

    if m < 98:
        # print('No data')
        nodata = nodata + 1
        nodata_address = '/Users/mateus.miranda/Documents/INPE-CAP/CERRA.NET/cerraNet_v3/data_preprocessing/images/nodata'
        shutil.move(full_address, nodata_address)
    else:
        # print('It has data')
        hasdata = hasdata + 1

print('In this dir has ', nodata, ' images no data, and', hasdata, 'images with data.')


