import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tqdm import tqdm
import requests
import json
import os
import time
import pandas as pd
import numpy as np
from PIL import Image
import geopandas as gpd
import geojson
import shapely
import rasterio as rio
from rasterio.plot import show
import rasterio.mask
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ee
import eeconvert as eec
import geemap
from geemap.foliumap import Map as emap
import folium
import urllib.request
import zipfile
import tempfile
import shutil
import torch
from torchvision import datasets, models, transforms

# earthengine initalization 
try:
    ee.Initialize(project="land-sat-458621")
    print("earthengine initialized successfully!")
except Exception as e:
    print("earthengine authenticate") 
    # download google cloud SDK, and run "gcloud auth" in CLI
    # run "earthengine authenticate" in CLI

# set country & administrative level ()
ISO = 'IND' # "IND" is the ISO code for India
ADM = 'ADM3' # equivalent to administrative districts

# download country geoBoundaries
r = requests.get("https://www.geoboundaries.org/api/current/gbOpen/{}/{}/".format(ISO, ADM))
r.raise_for_status()
data = r.json()
print(type(data), data)

# handle API response -> if dict, turn into list for iteration
if isinstance(data, dict):
    entries = [data]
elif isinstance(data, list):
    entries = data
else:
    raise ValueError(f"Unexpected response type: {type(data)}")

dl_path = entries[0]['gjDownloadURL'] #download url for geoJSON file

# Save the result as a geoJSON
filename = 'geoboundary.geojson'
geoboundary = requests.get(dl_path).json()
with open(filename, 'w') as file:
   geojson.dump(geoboundary, file)

# read geoJSON data using geopandas
geoboundary = gpd.read_file(filename)
print("Data dimensions: {}".format(geoboundary.shape))
geoboundary.sample(10) # show 10 sample areas
shape_name = 'Sarita Vihar' # find specific area
fig, ax = plt.subplots(1, figsize=(10,10))
geoboundary[geoboundary.shapeName == shape_name].plot('shapeName', legend=True, ax=ax)

# to get satellite colour (3 band RGB) image for a region
def generate_image(
    region,  # geojson?
    product='COPERNICUS/S2', # sentinel-2
    min_date='2018-01-01', # start date
    max_date='2020-01-01', # end date
    range_min=0,
    range_max=2000,
    cloud_pct=10 # cloud coverage percentage
):

    # gets collection of S2 images & filters them
    image = ee.ImageCollection(product)\
        .filterBounds(region)\
        .filterDate(str(min_date), str(max_date))\
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))\
        .median() # combines all filtered images into single image -> median value for each pixel

    image = image.visualize(bands=['B4', 'B3', 'B2'], min=range_min, max=range_max)  #get RGB bands
    # Note that the max value of the RGB bands is set to 65535
    # because the bands of Sentinel-2 are 16-bit integers
    # with a full numerical range of [0, 65535] (max is 2^16 - 1);
    # however, the actual values are much smaller than the max value.
    # Source: https://stackoverflow.com/a/63912278/4777141

    return image.clip(region)

region = geoboundary.loc[geoboundary['shapeName'] == shape_name] # selects area of interest
region = region.to_crs("EPSG:3857") #???
centroid = region.geometry.centroid.iloc[0].coords[0] # finds center of region (for map display)
region_fc = eec.gdfToFc(region) # coverts to google ee usable format

# get satellite image for AOI & year 
image = generate_image(
    region_fc,
    product='COPERNICUS/S2',
    min_date='2020-01-01',
    max_date='2020-12-31',
    cloud_pct=10,
)

# create interactive map & display region
m = emap(center=(centroid[1], centroid[0]), zoom=10)
m.addLayer(image,{}, 'Sentinel-2')
m.addLayerControl()
m.save("sentinel2_map.html")  # renders the interactive map webpage


def export_image(image, filename, region, folder):
    
    # convert GeoDataFrame to ee geometry
    region_fc = geemap.gdf_to_ee(region) 
    region_geometry = region_fc.geometry()

    print(f'Exporting {filename}.tif to {folder}...')
        
    # create output directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
        
    # set download parameters
    params = {
        'scale': 10, # pixel size
        'region': region_geometry, # area
        'crs': 'EPSG:4326', # coordinate system
        'format': 'GEO_TIFF', # file type
        'filePerBand': False # saving each band as a seperate file
    }

    # get download URL and download
    try:
        url = image.getDownloadURL(params)
        print(f"Downloading from URL: {url}")
            
        # saving file in folder
        output_path = os.path.join(folder, f"{filename}.tif")
        urllib.request.urlretrieve(url, output_path)
        print(f"Successfully saved to {output_path}")

    except Exception as e: # error handling
        print(f"Export failed: {str(e)}")
        raise


cwd = './data/'  # local data directory

# create necessary directories
input_dir = os.path.join(cwd, 'input/')  # create input subdirectory
sample_predictions_dir = './sample_predictions/'  # directory for predictions
os.makedirs(input_dir, exist_ok=True)
os.makedirs(sample_predictions_dir, exist_ok=True)

tif_file = os.path.join(input_dir, f'{shape_name}.tif') #tif input file path
export_image(image, shape_name, region, input_dir) # export to input subdirectory

# open image file using Rasterio
image = rio.open(tif_file)
boundary = geoboundary[geoboundary.shapeName == shape_name]

# plot image and corresponding boundary
fig, ax = plt.subplots(figsize=(15,15))
boundary.plot(facecolor="none", edgecolor='red', ax=ax)
show(image, ax=ax) 

# divides image into small tiles/patches
def generate_tiles(image_file, output_file, area_str, size=64):
    #open the raster image using rasterio
    raster = rio.open(image_file)
    width, height = raster.shape

    # create a dictionary which will contain 64x64 polygon tiles
    geo_dict = { 'id' : [], 'geometry' : []}
    index = 0

    # calculate total number of tiles
    total_tiles = ((width + size - 1) // size) * ((height + size - 1) // size)

    # do a sliding window across the raster image -> in steps of size (64x64)
    with tqdm(total=total_tiles, desc="Generating tiles") as pbar:
        for w in range(0, width, size):
            for h in range(0, height, size):
                window = rio.windows.Window(h, w, size, size) # create Window of your desired size
                bbox = rio.windows.bounds(window, raster.transform) # get georeferenced window bounds
                bbox = box(*bbox) # turns coordinates into polygon object
                
                uid = '{}-{}'.format(area_str.lower().replace(' ', '_'), index) # create unique id for each geometry -> sarita_vihar-0
                geo_dict['id'].append(uid) # update dictionary
                geo_dict['geometry'].append(bbox)

                index += 1
                pbar.update(1)  # Update by 1 for each tile processed

    results = gpd.GeoDataFrame(pd.DataFrame(geo_dict)) # convert dict to geopandas dataframe
    results.crs = {'init' :'epsg:4326'} # set coordinate ref system -> standard latitude/longitude system (EPSG:4326)
    results.to_file(output_file, driver="GeoJSON") # save file as GeoJSON

    raster.close() # close image file
    return results

output_file = input_dir+'{}.geojson'.format(shape_name) # file for tiles
tiles = generate_tiles(tif_file, output_file, shape_name, size=64) # generates tiles 64x64
print('Data dimensions: {}'.format(tiles.shape))
tiles.head(3)

image = rio.open(tif_file)
fig, ax = plt.subplots(figsize=(15,15))
tiles.plot(facecolor="none", edgecolor='red', ax=ax)
show(image, ax=ax)


image = rio.open(tif_file)
tiles = tiles.to_crs(boundary.crs) # ensure tiles & boundary use same CRS (coordinate ref system)
for df in (tiles, boundary):
    if 'index_right' in df.columns: # removes right column to avoid errors in next job
        df.drop(columns=['index_right'], inplace=True)

# spatial join with -> finds all tiles within boundary polygon 
tiles_within = gpd.sjoin(tiles, boundary, how="inner", predicate="within")

# plot the tiles
fig, ax = plt.subplots(figsize=(15, 15))
tiles_within.plot(facecolor="none", edgecolor="red", ax=ax)
show(image, ax=ax)

def show_crop(image, shape, title='', save_path=None):
    with rio.open(image) as src: # opens satellite image
        out_image, out_transform = rio.mask.mask(src, shape, crop=True) #mask & crop image to shape
        _, x_nonzero, y_nonzero = np.nonzero(out_image) # find non zero pixels -> pixels that have data
      
      # check if x_nonzero and y_nonzero are empty
        if x_nonzero.size == 0 or y_nonzero.size == 0:
            print("Warning: Tile does not intersect with valid image data.")
            return # skip this tile
        out_image = out_image[
            :,
            np.min(x_nonzero):np.max(x_nonzero),
            np.min(y_nonzero):np.max(y_nonzero)
        ] # crop out any black border -> leaving only data
        
        # update metadata and save the cropped image
        if save_path is not None:
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width":  out_image.shape[2],
                "transform": out_transform
            })
            with rio.open(save_path, 'w', **out_meta) as dest:
                dest.write(out_image)
        
        # visualize the cropped image
        # show(out_image, title=title)
        return out_image

# show_crop(tif_file, [tiles.iloc[5]['geometry']]) # takes 5th image, crops image to just the tile, displays it

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
    inference_times = [] # track inference time of each grid
    for shape in shapes:
        temp_dir = tempfile.gettempdir()
        temp_tif = os.path.join(temp_dir, "temp_crop.tif")
        show_crop(image_path, [shape], save_path=temp_tif)
        
        # run through your torchgeo model
        img = Image.open(temp_tif)
        inp = transform(img) # apply existing transform()
        
        start_time = time.time() # measure inference time
        out = model(inp.unsqueeze(0)) # runs image through model 
        # unsqueeze(0) -> adds a new dimension to a tensor: [channels, height, weight] -> [batch_size, channels, height, width]
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
       
        _, pred = torch.max(out, 1) # finds predicted class -> max probability
        label = classes[int(pred[0])] # gets predicted label class

        if show:
            img.show(title=label)

        return label, inference_time
    return None, None

# commence model prediction
labels = [] # list to store predictions
inference_times = [] # list to store inference times
total_start_time = time.time()

for index in tqdm(range(len(tiles)), total=len(tiles)):
    label, inference_time = predict_crop(tif_file, [tiles.iloc[index]['geometry']], classes, model)
    labels.append(label)
    inference_times.append(inference_time)

total_time = time.time() - total_start_time
avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

print(f"total processing time: {total_time:.2f} seconds")
print(f"average inference time per image: {avg_inference_time*1000:.2f} ms")
print(f"images per second: {len(tiles)/total_time:.2f}")

tiles['pred'] = labels
tiles['inference_time'] = inference_times

# save predictions
filepath = os.path.join(sample_predictions_dir, f"{shape_name}_preds.geojson")
tiles.to_file(filepath, driver="GeoJSON")
tiles.head(3)

filepath = os.path.join(sample_predictions_dir, f"{shape_name}_preds.geojson") # read predictions 
tiles = gpd.read_file(filepath) 
tiles = tiles.to_crs("EPSG:4326") # Convert to WGS84 (EPSG:4326) for folium compatibility (for better mapping)
tiles.geometry = tiles.geometry.make_valid() # ensure geometries are valid

# debug checks
print("Unique predictions:", tiles['pred'].unique())
print("Prediction counts:\n", tiles['pred'].value_counts())

# color mapping for classes
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

tiles['color'] = tiles['pred'].map(colors).fillna('#000000') # create color column with black as default
tiles['color'] = tiles['color'].apply(lambda x: mcolors.to_hex(x) if isinstance(x, str) else x) # convert color names to hex (if using named colors)

region = region.to_crs("EPSG:4326")
centroid = region.geometry.centroid.iloc[0].coords[0] # get valid centroid in WGS84

# create folium map with proper zoom
m = folium.Map(
    location=[centroid[1], centroid[0]],  # folium expects [lat, lon]
    zoom_start=12,
    tiles='CartoDB positron'  # more reliable default
)

# add google satellite (if available)
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr='Google',
    name='Satellite',
    overlay=False,
    control=True
).add_to(m)

# create feature group for predictions
fg = folium.FeatureGroup(name='Land Cover', show=True)

# add each tile to the map with its colour
for _, row in tiles.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda x, color=row['color']: { 
            'fillColor': color, # colour
            'color': 'black', # outline
            'weight': 0.5, # thickness
            'fillOpacity': 0.7 # opacity
        }
    ).add_to(fg)
fg.add_to(m) # add feature group to map
folium.LayerControl().add_to(m) # add layer control and legend

# create legend
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

# save map 
output_map = os.path.join(sample_predictions_dir, f"{shape_name}_map.html")
m.save(output_map)
print(f"Map saved to: {output_map}")