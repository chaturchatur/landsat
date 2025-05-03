import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# Standard imports
from tqdm.notebook import tqdm
import requests
import json
import os
import time

import pandas as pd
import numpy as np
from PIL import Image

# Geospatial processing packages
import geopandas as gpd
import geojson

import shapely
import rasterio as rio
from rasterio.plot import show
import rasterio.mask
from shapely.geometry import box

# Mapping and plotting libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ee
import eeconvert as eec
import geemap
# NEW
from geemap.foliumap import Map as emap
import folium
import urllib.request
import zipfile
import tempfile
import shutil

# Deep learning libraries
import torch
from torchvision import datasets, models, transforms


try:
    ee.Initialize(project="land-sat-458621")
    print("Earth Engine initialized successfully!")
    print("\nCurrent Earth Engine user info:")
except Exception as e:
    print("Please authenticate Earth Engine first by running:")
    print("earthengine authenticate")

ISO = 'IND' # "DEU" is the ISO code for Germany
ADM = 'ADM3' # Equivalent to administrative districts

# Query geoBoundaries
r = requests.get("https://www.geoboundaries.org/api/current/gbOpen/{}/{}/".format(ISO, ADM))
r.raise_for_status()
data = r.json()
print(type(data), data)

if isinstance(data, dict):
    entries = [data]
elif isinstance(data, list):
    entries = data
else:
    raise ValueError(f"Unexpected response type: {type(data)}")

dl_path = entries[0]['gjDownloadURL']

# Save the result as a GeoJSON
filename = 'geoboundary.geojson'
geoboundary = requests.get(dl_path).json()
with open(filename, 'w') as file:
   geojson.dump(geoboundary, file)

# Read data using GeoPandas
geoboundary = gpd.read_file(filename)
print("Data dimensions: {}".format(geoboundary.shape))
vasant_kunj = geoboundary[geoboundary['shapeName'].str.contains('Vasant Kunj', case=False)]
geoboundary.sample(10)

shape_name = 'Sarita Vihar'
fig, ax = plt.subplots(1, figsize=(10,10))
geoboundary[geoboundary.shapeName == shape_name].plot('shapeName', legend=True, ax=ax);

def generate_image(
    region,
    product='COPERNICUS/S2',
    min_date='2018-01-01',
    max_date='2020-01-01',
    range_min=0,
    range_max=2000,
    cloud_pct=10
):
    # Generate median aggregated composite
    image = ee.ImageCollection(product)\
        .filterBounds(region)\
        .filterDate(str(min_date), str(max_date))\
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))\
        .median()

    # Get RGB bands
    image = image.visualize(bands=['B4', 'B3', 'B2'], min=range_min, max=range_max)
    # Note that the max value of the RGB bands is set to 65535
    # because the bands of Sentinel-2 are 16-bit integers
    # with a full numerical range of [0, 65535] (max is 2^16 - 1);
    # however, the actual values are much smaller than the max value.
    # Source: https://stackoverflow.com/a/63912278/4777141

    return image.clip(region)

region = geoboundary.loc[geoboundary['shapeName'] == shape_name]
region = region.to_crs("EPSG:3857")
centroid = region.geometry.centroid.iloc[0].coords[0]
region_fc = eec.gdfToFc(region)

# 3. Build a Sentinel-2 Surface Reflectance RGB composite for 2021
image = generate_image(
    region_fc,
    product='COPERNICUS/S2',
    min_date='2020-01-01',
    max_date='2020-12-31',
    cloud_pct=10,
)

m = emap(center=(centroid[1], centroid[0]), zoom=10)
m.addLayer(image,{}, 'Sentinel-2')
m.addLayerControl()
m  # renders the interactive map in the notebook

def export_image(image, filename, region, folder):
    try:
        # Initialize with your new project
        ee.Initialize(project="land-sat-458621")
        
        # Convert GeoDataFrame to EE geometry
        region_fc = geemap.gdf_to_ee(region)
        region_geometry = region_fc.geometry()

        print(f'Exporting {filename}.tif to {folder}...')
        
        # Create output directory if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Get download parameters
        params = {
            'scale': 10,
            'region': region_geometry,
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF',
            'filePerBand': False
        }

        # Get download URL and download
        try:
            url = image.getDownloadURL(params)
            print(f"Downloading from URL: {url}")
            
            # Download the file directly
            output_path = os.path.join(folder, f"{filename}.tif")
            urllib.request.urlretrieve(url, output_path)
            print(f"Successfully saved to {output_path}")

        except Exception as e:
            print(f"Export failed: {str(e)}")
            raise

    except Exception as e:
        print(f"Earth Engine initialization failed: {str(e)}")
        print("Please make sure you have:")
        print("1. Created a Google Cloud project")
        print("2. Enabled the Earth Engine API")
        print("3. Authenticated with Earth Engine")
        raise

    

# Change this to your local directory
cwd = './data/'  # Local data directory
input_dir = os.path.join(cwd, 'input/')  # Create input subdirectory
sample_predictions_dir = './sample_predictions/'  # Directory for predictions

# Create necessary directories
os.makedirs(input_dir, exist_ok=True)
os.makedirs(sample_predictions_dir, exist_ok=True)

# File paths
tif_file = os.path.join(input_dir, f'{shape_name}.tif')

# Export the image
export_image(image, shape_name, region, input_dir)

# Open image file using Rasterio
image = rio.open(tif_file)
boundary = geoboundary[geoboundary.shapeName == shape_name]

# Plot image and corresponding boundary
fig, ax = plt.subplots(figsize=(15,15))
boundary.plot(facecolor="none", edgecolor='red', ax=ax)
show(image, ax=ax);

def generate_tiles(image_file, output_file, area_str, size=64):
    # Open the raster image using rasterio
    raster = rio.open(image_file)
    width, height = raster.shape

    # Create a dictionary which will contain our 64 x 64 px polygon tiles
    # Later we'll convert this dict into a GeoPandas DataFrame.
    geo_dict = { 'id' : [], 'geometry' : []}
    index = 0

    # Do a sliding window across the raster image
    with tqdm(total=width*height) as pbar:
      for w in range(0, width, size):
          for h in range(0, height, size):
              # Create a Window of your desired size
              window = rio.windows.Window(h, w, size, size)
              # Get the georeferenced window bounds
              bbox = rio.windows.bounds(window, raster.transform)
              # Create a shapely geometry from the bounding box
              bbox = box(*bbox)

              # Create a unique id for each geometry
              uid = '{}-{}'.format(area_str.lower().replace(' ', '_'), index)

              # Update dictionary
              geo_dict['id'].append(uid)
              geo_dict['geometry'].append(bbox)

              index += 1
              pbar.update(size*size)

    # Cast dictionary as a GeoPandas DataFrame
    results = gpd.GeoDataFrame(pd.DataFrame(geo_dict))
    # Set CRS to EPSG:4326
    results.crs = {'init' :'epsg:4326'}
    # Save file as GeoJSON
    results.to_file(output_file, driver="GeoJSON")

    raster.close()
    return results

output_file = input_dir+'{}.geojson'.format(shape_name)
tiles = generate_tiles(tif_file, output_file, shape_name, size=64)
print('Data dimensions: {}'.format(tiles.shape))
tiles.head(3)

image = rio.open(tif_file)
fig, ax = plt.subplots(figsize=(15,15))
tiles.plot(facecolor="none", edgecolor='red', ax=ax)
show(image, ax=ax);

# 1. Open your GeoTIFF
image = rio.open(tif_file)

# 2. Make sure both GeoDataFrames share the same CRS
tiles = tiles.to_crs(boundary.crs)

# 3. Drop any pre‐existing 'index_right' column to avoid name collisions
for df in (tiles, boundary):
    if 'index_right' in df.columns:
        df.drop(columns=['index_right'], inplace=True)

# 4. Perform the spatial join with the new `predicate` argument
tiles_within = gpd.sjoin(tiles, boundary, how="inner", predicate="within")

# 5. Plot the result
fig, ax = plt.subplots(figsize=(15, 15))
tiles_within.plot(facecolor="none", edgecolor="red", ax=ax)
show(image, ax=ax)

def show_crop(image, shape, title=''):
  with rio.open(image) as src:
      out_image, out_transform = rio.mask.mask(src, shape, crop=True)
      # Crop out black (zero) border
      _, x_nonzero, y_nonzero = np.nonzero(out_image)
      
      # Check if x_nonzero and y_nonzero are empty
      if x_nonzero.size == 0 or y_nonzero.size == 0:
          print("Warning: Tile does not intersect with valid image data.")
          return # Skip this tile
      out_image = out_image[
        :,
        np.min(x_nonzero):np.max(x_nonzero),
        np.min(y_nonzero):np.max(y_nonzero)
      ]
      # Visualize image
      show(out_image, title=title)

show_crop(tif_file, [tiles.iloc[5]['geometry']])

# LULC Classes
classes = [
  'AnnualCrop',
  'Forest',
  'HerbaceousVegetation',
  'Highway',
  'Industrial',
  'Pasture',
  'PermanentCrop',
  'Residential',
  'River',
  'SeaLake'
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_file = './models/best_model_torchgeo.pth'

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

print('Model file {} successfully loaded.'.format(model_file))

imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

def predict_crop(image_path, shapes, classes, model, show=False):
    """Generates model prediction using trained model, but skips zero‐size crops."""
    for shape in shapes:
        with rio.open(image_path) as src:
            # 1. Crop source image using the polygon shape
            out_image, out_transform = rio.mask.mask(src, [shape], crop=True)

            # 2. Find non-zero pixels
            _, x_nonzero, y_nonzero = np.nonzero(out_image)
            if x_nonzero.size == 0 or y_nonzero.size == 0:
                # nothing intersected this tile — skip it
                return None

            # 3. Trim out zero‐border around the crop
            out_image = out_image[
                :,
                x_nonzero.min(): x_nonzero.max() + 1,
                y_nonzero.min(): y_nonzero.max() + 1
            ]

            # 4. Update metadata for the cropped patch
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width":  out_image.shape[2],
                "transform": out_transform
            })

            # 5. Write to a temporary file
            temp_tif = '/tmp/temp_crop.tif'
            with rio.open(temp_tif, 'w', **out_meta) as dest:
                dest.write(out_image)

        # 6. Run through your PyTorch model
        img = Image.open(temp_tif)
        inp = transform(img)               # your existing transform()
        out = model(inp.unsqueeze(0))
        _, pred = torch.max(out, 1)
        label = classes[int(pred[0])]

        if show:
            img.show(title=label)

        return label

    return None

# Commence model prediction
labels = [] # Store predictions
for index in tqdm(range(len(tiles)), total=len(tiles)):
  label = predict_crop(tif_file, [tiles.iloc[index]['geometry']], classes, model)
  labels.append(label)
tiles['pred'] = labels

# Save predictions
filepath = os.path.join(sample_predictions_dir, f"{shape_name}_preds.geojson")
tiles.to_file(filepath, driver="GeoJSON")
tiles.head(3)

# Read predictions
filepath = os.path.join(sample_predictions_dir, f"{shape_name}_preds.geojson")
tiles = gpd.read_file(filepath)

# Convert to WGS84 (EPSG:4326) for Folium compatibility
tiles = tiles.to_crs("EPSG:4326")

# Ensure geometries are valid
tiles.geometry = tiles.geometry.make_valid()

# Add debug checks
print("Unique predictions:", tiles['pred'].unique())
print("Prediction counts:\n", tiles['pred'].value_counts())

# Color mapping with fallback for missing classes
colors = {
    'AnnualCrop': 'lightgreen',
    'Forest': 'forestgreen',
    'HerbaceousVegetation': 'yellowgreen',
    'Highway': 'gray',
    'Industrial': 'red',
    'Pasture': 'mediumseagreen',
    'PermanentCrop': 'chartreuse',
    'Residential': 'magenta',
    'River': 'dodgerblue',
    'SeaLake': 'blue'
}

# Create color column with black as default
tiles['color'] = tiles['pred'].map(colors).fillna('#000000')

# Convert color names to hex (if using named colors)
tiles['color'] = tiles['color'].apply(lambda x: mcolors.to_hex(x) if isinstance(x, str) else x)

# Get valid centroid in WGS84
region = region.to_crs("EPSG:4326")
centroid = region.geometry.centroid.iloc[0].coords[0]

# Create map with proper zoom
m = folium.Map(
    location=[centroid[1], centroid[0]],  # Folium expects [lat, lon]
    zoom_start=12,
    tiles='CartoDB positron'  # More reliable default
)

# Add Google Satellite (if available)
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr='Google',
    name='Satellite',
    overlay=False,
    control=True
).add_to(m)

# Create feature group for predictions
fg = folium.FeatureGroup(name='Land Cover', show=True)

# Add geometries with proper style binding
for _, row in tiles.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda x, color=row['color']: {
            'fillColor': color,
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7
        }
    ).add_to(fg)

fg.add_to(m)

# Add layer control and legend
folium.LayerControl().add_to(m)

# Create legend
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 180px; 
     background-color: white; padding: 10px;
     border:2px solid grey; z-index:9999;
     font-size:14px;">
     <b>Land Cover Legend</b><br>
'''
for label, color in colors.items():
    legend_html += f'<i style="background:{color}; width:20px; height:20px; display:inline-block;"></i> {label}<br>'

legend_html += '</div>'
m.get_root().html.add_child(folium.Element(legend_html))

# Save and display
output_map = os.path.join(sample_predictions_dir, f"{shape_name}_map.html")
m.save(output_map)
print(f"Map saved to: {output_map}")

# Open in default browser
import webbrowser
webbrowser.open(output_map)