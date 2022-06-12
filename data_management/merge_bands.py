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
import earthpy.spatial as es
import earthpy.plot as ep
from osgeo import gdal

# Data
import pandas as pd
import numpy as np


# -------- ACCESS THE DATA
# Constants
image = []
df = []
data = []

# Reading every file in each one subdirectory
for path in glob.iglob('data/raw/**'):
    # Read each image at the respectively subdirectory
    for band in glob.iglob(path + '/*.tif'):
        # Save each one image
        image.append(band)
    # Get the 5 last bands for each iteration in order
    data.append(sorted(image[-5:]))

# Build a dataframe from data
df = pd.DataFrame(data)

# Save all in .csv
#df.to_csv('df1.csv', sep='\t', encoding='utf-8')

# -------- MERGE BANDS
# Create the diretory for saving the news files
output_dir = os.path.join("data_preprocessing/images_composed", "tocantins")
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)

for j in range(len(df)):
    # Selecting the bands NIR, Green, and Blue
    bands = [df[4][j], df[2][j], df[1][j]]

    # Saving the images composed
    name_raster = os.path.basename(df[0][j])
    raster_out_path = os.path.join(output_dir, name_raster)

    # Create image stack
    arr_st, meta = es.stack(bands, out_path=raster_out_path)


