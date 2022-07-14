import geopandas as gpd
import pandas as pd
import csv

for i in range(1,11):
    source = pd.read_csv('mockaroo/mockaroo_points/mockaroo{}.csv'.format(i))
    source_gdf = gpd.GeoDataFrame(source, geometry=gpd.points_from_xy(source.lon, source.lat))
    source_gdf.to_file('mockaroo/mockaroo_points/mockaroo{}.geojson'.format(i), driver='GeoJSON')


