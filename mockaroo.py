import geopandas as gpd
import pandas as pd
import csv

def mockaroo_geojson():
    for i in range(1,11):
        source = pd.read_csv('mockaroo/mockaroo_points/mockaroo{}.csv'.format(i))
        source_gdf = gpd.GeoDataFrame(source, geometry=gpd.points_from_xy(source.lon, source.lat))
        source_gdf.to_file('mockaroo/mockaroo_points/mockaroo{}.geojson'.format(i), driver='GeoJSON')

filename = "geojson_polygons/usa.geojson"

source = gpd.read_file(filename)
source = source.to_crs(epsg=2163)
print("Triangle area = " + str(round(source.area[0]/1000000, 2)) + "km^2")
