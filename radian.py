# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:54:29 2022

@author: paddy
"""
# Package imports
import random
from random import randint
import time

import os
import json
import string
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
import warnings

# voroni generation packages
from shapely.ops import cascaded_union
from geovoronoi import voronoi_regions_from_coords, points_to_coords

# k-means clustering packages
from sklearn.cluster import KMeans

# misc json and shapely packages
from geojson import Feature, FeatureCollection, Point
import shapely
from shapely.geometry import Polygon, Point, shape, GeometryCollection, LineString, LinearRing
from shapely.geometry import box

# This function takes in a Shapely Polygon object and returns a GeoDataFrame consisting of Voronoi-based buffers
# centred around either the true centroid of the original Polygon, or around a randomly generated "moving centroid"
# The function has three different forms of generation:
# 'eq' = Equal-area Voronoi generation centred around the original polygon centroid
# 'area' = Variable-area generation (Smaller Voronoi towards the centroid, larger towards the borders)
# 'rand' = Equal-area Voronoi generation centred around a random "moving centroid"
global global_accepted_points
global global_rejected_points
global glob_ratio_list

global_accepted_points = 0
global_rejected_points = 0
glob_ratio_list = []

# Suppress depreciation warnings
warnings.filterwarnings('ignore')

def poly_bb_ratio(poly):
    min_x, min_y, max_x, max_y = poly.bounds
    bb = gpd.GeoSeries(box(min_x, min_y, max_x, max_y, ccw=True))
    ratio = float(1/(poly.area/bb.area))
    if ratio < 2:
        return 2
    return ratio

########## POINT GENERATION ##########

def points_uniform(poly, num_points):
    global global_rejected_points
    global global_accepted_points
    min_x, min_y, max_x, max_y = poly.bounds
    poly_ratio = poly_bb_ratio(poly)
    poly_gdf = gpd.GeoDataFrame(pd.DataFrame([poly], columns=['geometry']), geometry='geometry', crs=4326)

    points = []
    # Generates points repeatedly with a uniform generation within the bounds of the polygon
    while len(points) < round(num_points * poly_ratio):
        points.append(Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)]))

    gdf = gpd.GeoDataFrame(pd.DataFrame(points, columns=['geometry']), geometry='geometry', crs=4326)
    gdf = gdf.sjoin(poly_gdf, predicate='within')
    gdf = gdf.drop(['index_right'], axis=1)

    return gdf.iloc[0:num_points].reset_index(drop=True)
    
    #return gpd.GeoDataFrame(pd.DataFrame([], columns=['geometry']), geometry='geometry')

def points_moving_centre(poly, num_points):
    global global_rejected_points
    global global_accepted_points
    min_x, min_y, max_x, max_y = poly.bounds
    cx, cy = poly.centroid.x, poly.centroid.y
    max_pt = Point(max_x, max_y)
    radius = max_pt.distance(poly.centroid)

    #poly_ratio = poly_bb_ratio(poly)
    poly_gdf = gpd.GeoDataFrame(pd.DataFrame([poly], columns=['geometry']), geometry='geometry')

    points = []
    # Moving centroid is generated in an eliptical region around the original centroid
    range_x = (max_x - min_x) / 4
    range_y = (max_y - min_y) / 16
    centroid_point = Point([random.uniform(cx - range_x, cx + range_x), random.uniform(cy - range_y, cy + range_y)])

    # Number of sections is set as well as the number of points assigned to each section
    section_num = 5
    section_size = (radius * 0.8) / section_num
    section_pts = round(num_points / section_num)

    # Values used to shift point locations to account for the moving centroid generation
    cent_diff_x = (cx - centroid_point.x) / section_num
    cent_diff_y = (cy - centroid_point.y) / section_num

    points_gdf = gpd.GeoDataFrame([])
    # Points are generated section by section
    for i in range(0, section_num):
        if num_points % 5 != 0:
            if i == 4:
                temp = section_pts * i
                section_pts = num_points - temp
        # Current circular buffer is created within which to generate points
        point_current = Point([centroid_point.x + (cent_diff_x * i), centroid_point.y + (cent_diff_y * i)])
        c_current = point_current.buffer(section_size * (i + 1))
        min_x, min_y, max_x, max_y = c_current.bounds

        buffer_gdf = gpd.GeoDataFrame(pd.DataFrame([c_current], columns=['geometry']), geometry='geometry')
        buffer_ratio = poly_bb_ratio(c_current)

        # This while loop controls the generation of points in the current section
        current_list = []
        while len(current_list) < section_pts * buffer_ratio:
            # here we generate a point using a uniform distribution to set the possible x and y ranges
            current_list.append(Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)]))

        gdf = gpd.GeoDataFrame(pd.DataFrame(current_list, columns=['geometry']), geometry='geometry')
        gdf = gdf.sjoin(buffer_gdf, predicate='within')
        gdf = gdf.drop(['index_right'], axis=1)
        gdf = gdf.sjoin(poly_gdf, predicate='within')
        gdf = gdf.drop(['index_right'], axis=1)

        points_gdf = pd.concat([points_gdf, gdf],ignore_index=True)


    # Points list is converted to a GeoDataFrame and outputted
    #df = pd.DataFrame(points, columns=['geometry'])
    #gdf = gpd.GeoDataFrame(df, geometry='geometry')
    return points_gdf.iloc[0:num_points].reset_index(drop=True)

def points_centre(poly, num_points):
    global global_rejected_points
    global global_accepted_points
    min_x, min_y, max_x, max_y = poly.bounds
    poly_gdf = gpd.GeoDataFrame(pd.DataFrame([poly], columns=['geometry']), geometry='geometry')

    cx, cy = poly.centroid.x, poly.centroid.y
    max_pt = Point(max_x, max_y)
    radius = max_pt.distance(poly.centroid)

    points = []
    
    section_num = 5
    section_size = (radius * 0.8) / section_num
    section_pts = round(num_points / section_num)
    points_gdf = gpd.GeoDataFrame([])
    for i in range(0, section_num):
        if num_points % 5 != 0:
            if i == 4:
                temp = section_pts * i
                section_pts = num_points - temp
        c_current = poly.centroid.buffer(section_size * (i+1))
        min_x, min_y, max_x, max_y = c_current.bounds
        buffer_gdf = gpd.GeoDataFrame(pd.DataFrame([c_current], columns=['geometry']), geometry='geometry')
        current_list = []
        while len(current_list) < section_pts*3:
            current_list.append(Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)]))
            
        gdf = gpd.GeoDataFrame(pd.DataFrame(current_list, columns=['geometry']), geometry='geometry')
        gdf = gdf.sjoin(buffer_gdf, predicate='within')
        gdf = gdf.drop(['index_right'], axis=1)
        gdf = gdf.sjoin(poly_gdf, predicate='within')
        gdf = gdf.drop(['index_right'], axis=1)

        points_gdf = pd.concat([points_gdf, gdf], ignore_index=True)

    #df = pd.DataFrame(points, columns=['geometry'])
    #gdf = gpd.GeoDataFrame(df, geometry='geometry')

    return points_gdf.iloc[0:num_points].reset_index(drop=True)

########## VORONOI POLYGON GENERATION ##########

def kmeans_centroids(poly, num_points, num_cluster, eq_area):
    # Points are generated randomly in the polygon
    if(eq_area == 0): # Uniform distribution
        source = points_uniform(poly, num_points)  
    else: # Centroid-focused distribution
        source = points_centre(poly, num_points)

    # The geometries of the Shapely points are converted to a numpy array for use in the kmeans algorithm
    feature_coords = np.array([[e.x, e.y] for e in source.geometry])

    # A kmeans object is created using the specified number of clusters
    kmeans = KMeans(num_cluster, random_state=glob_random_seed)
    kmeans.fit(feature_coords)

    # The cluster centres are stored as centroids, and this list is put into a GeoDataFrame and returned
    centroids = kmeans.cluster_centers_
    df = pd.DataFrame(centroids, columns=['x', 'y'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

    return gdf

def moving_centroid(poly, epsg):
    cx, cy = poly.centroid.x, poly.centroid.y
    min_x, min_y, max_x, max_y = poly.bounds

    range_x = (max_x - min_x) / 4
    range_y = (max_y - min_y) / 16

    centroid_gdf = gpd.GeoDataFrame(pd.DataFrame([Point([random.uniform(cx - range_x, cx + range_x), random.uniform(cy - range_y, cy + range_y)])], columns=['geometry']), geometry='geometry', crs=3857)

    return centroid_gdf

def voronoi_gen(poly, poly_centroid, vor_num, gen_type):
    # Voronoi centroids are generated based on the specified generation type
    if(gen_type == 'eq'): # Equal-area uniformly distributed Voronoi regions
        vor_centroids = kmeans_centroids(poly, 500, vor_num, 0)
    elif(gen_type == 'area'): # Variable area, centrally focused Voronoi regions
        vor_centroids = kmeans_centroids(poly, 500, vor_num, 1)
    elif(gen_type == 'rand'): # Equal-area uniformly distributed Voronoi regions (with moving centroid)
        # Calculate moving centroid in an eliptical region around the original centroid
        boundary_poly = poly#['geometry'][0]

        gdf_centroid = poly_centroid
        #gdf_centroid.crs = poly.crs

        vor_centroids = kmeans_centroids(poly, 500, vor_num, 0)
    # Setting crs to meter based projection
    #gdf_proj = vor_centroids.set_crs(poly.crs)

    # Convert the boundary geometry into a union of the polygon
    boundary_shape = cascaded_union(poly)#.geometry)
    #coords = points_to_coords(gdf_proj.geometry)
    coords = points_to_coords(vor_centroids.geometry)

    # Calculating the voronoi regions
    region_polys, region_pts = voronoi_regions_from_coords(coords, boundary_shape)


    df = pd.DataFrame(list(region_polys.items()), columns=['index','geometry'])
    gdf_poly = gpd.GeoDataFrame(df, geometry='geometry', crs=3857)
    #gdf_poly.crs = poly.crs

    # Calculating distance of Voronoi polygons to the centroid (moving or original)
    gdf_poly['dist_to_centre'] = 0
    for i in range(vor_num):
        if(gen_type == 'rand'):
            current = gdf_poly['geometry'][i].centroid.distance(gdf_centroid.iloc[0])
        else:
            current = gdf_poly['geometry'][i].centroid.distance(
            shapely.geometry.Point(poly.centroid.x, poly.centroid.y))
        gdf_poly['dist_to_centre'][i] = current

    # Assign a class to each polygon based on the distance to centroid
    # This will produce five distinct regions centred around the given moving/original centroid


    max_dist, min_dist = max(gdf_poly['dist_to_centre']), min(gdf_poly['dist_to_centre'])
    dist_break = (max_dist - min_dist) / 5
    gdf_poly['class'] = " "
    gdf_poly = gdf_poly.sort_values(by='dist_to_centre')
    gdf_poly['class'] = pd.cut(gdf_poly['dist_to_centre'], [0, dist_break, dist_break*2, dist_break*3, dist_break*4, np.inf], labels=[1,2,3,4,5])

    # Circular buffer visualization
    buffers = []
    title = "Voronoi-based Buffer Generation:\n"
    for i in range(5):
        if gen_type != 'rand':
            centroid = shapely.geometry.Point(poly.centroid.x, poly.centroid.y)
            c_current = centroid.buffer(dist_break * (i + 1))
            buffers.append(c_current)
        else:
            centroid = gdf_centroid#[0]
            c_current = centroid.buffer(dist_break * (i + 1))
            buffers.append(c_current)

    circ_df = pd.DataFrame(buffers, columns=['geometry'])
    circ_gdf = gpd.GeoDataFrame(circ_df, geometry='geometry')

    vor_union = gdf_poly.dissolve(by='class', as_index=False)

    return gdf_poly

########## ADDITIONAL METADATA FUNCTIONS

def get_var_type(var_type):
    if var_type == 'int':
        return "INTEGER"
    elif var_type == 'str':
        return "VARCHAR"
    elif var_type == 'ts':
        return "TIMESTAMP"


def gdf_poly_to_sql(table_name, gdf, directory):
    # initializes an SQL output file
    sqlFile = open(f'{directory}/SQL/{table_name}_voronoi.sql','w')
    sqlFile.write("")
    sqlFile.close()

    sqlFile = open(f'{directory}/SQL/{table_name}_voronoi.sql', 'a')
    sqlFile.write('-- Voronoi polygon regions exported to SQL from the RADIAN Spatal Data Generator\n\n')
    
    sqlFile.write('DROP TABLE IF EXISTS {}; \n\n'.format(table_name))
    sqlFile.write('CREATE TABLE {} ( \n'.format(table_name))
    sqlFile.write('\tpkid SERIAL PRIMARY KEY NOT NULL, \n')
    sqlFile.write("\tthegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326), \n")
    sqlFile.write("\tdist_to_centre NUMERIC,\n")
    sqlFile.write("\tpoly_class INTEGER\n")
    sqlFile.write('\n); \n\n')
    sqlFile.write('-- Spatial index is now created\n\n')
    # Creation of Spatial Index for the SQL file
    sqlFile.write('CREATE INDEX {}_spatial_index ON {} USING gist (thegeom); \n'.format(table_name, table_name))

    gdf['geometry'] = gdf.geometry.to_wkt()

    for row in gdf.itertuples():
        poly_coords = row[2]
        query = f"INSERT into {table_name} (thegeom, "
        query += f"{gdf.columns[2]}, poly_class"
        query += f") VALUES (ST_SetSRID(ST_PolygonFromText('{poly_coords}'),3857), "
        query += f"{row[1]}, {row[3]}); \r"
        sqlFile.write(query)

    # Write query string to SQL file
    print("Successfully export Voronoi polygons to SQL format.")


def gdf_to_sql(table_name, gdf, num_rows, random_vars, rand_var_types, rand_var_names, extra_var, extra_var_types, extra_var_name, extra_var_dict, directory):
    # Opens up an SQL file based on the table name, writes to the file and closes it
    sqlFile = open(f'{directory}/SQL/{table_name}.sql', "w")
    sqlFile.write("")
    sqlFile.close()

    # Opens up the SQL file to append lines to it
    sqlFile = open(f'{directory}/SQL/{table_name}.sql', "a")

    # SQL statments to create the table as well as drop if exists the table are appended
    sqlFile.write('-- This is an automatically generated SQL table. This has been generated by the RADIAN tool (developer Mr. Paddy Gorry)\n\n')

    sqlFile.write('DROP TABLE IF EXISTS {}; \n\n'.format(table_name))

    sqlFile.write('CREATE TABLE {} ( \n'.format(table_name))
    sqlFile.write('\tpkid SERIAL PRIMARY KEY NOT NULL, \n')
    sqlFile.write("\tthegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326)")

    if random_vars:
        sqlFile.write(',\n')
        for count, type in enumerate(rand_var_types):
            sqlFile.write(f'\t{rand_var_names[count]} {get_var_type(rand_var_types[count])}')
            if count < len(rand_var_types)-1:
                sqlFile.write(', \n')

    # "extra_var_dict": [{"type":"str", "name": "var_name", "source": "restaurant.csv"}]
    if extra_var:
        #for var_name in extra_var_name:
        #    create_query = ',\n\t{} '.format(var_name)
        #    if isinstance(gdf[f'{var_name}'][0], np.int64):
        #        create_query += 'INTEGER'
        #    else:
        #        create_query += 'VARCHAR'
        #    sqlFile.write(create_query)

        for variable in extra_var_dict:
            create_query = ',\n\t{} '.format(variable['name'])
            if isinstance(gdf[f"{variable['name']}"][0], np.int64):
                create_query += 'INTEGER'
            else:
                create_query += 'VARCHAR'
            sqlFile.write(create_query)

    sqlFile.write('\n); \n\n')

    sqlFile.write('-- Spatial index is now created\n\n')

    # Creation of Spatial Index for the SQL file
    sqlFile.write('CREATE INDEX {}_spatial_index ON {} USING gist (thegeom); \n'.format(table_name, table_name))

    # Loop through every point in the GeoDataFrame to write an instert statement to append to the SQL file
    for row in gdf.iterrows():
        # Set 'y' value to Latitude and 'x' value to Longitude.
        lat = row[1][0].y
        lon = row[1][0].x
        # Pull the randomly generated strings and ints from the dataframe
        if random_vars:
            if (not extra_var):
                query = f"INSERT into {table_name} (thegeom, "
                for count, type in enumerate(rand_var_names):
                    query += f"{rand_var_names[count]}"
                    if count < len(rand_var_names)-1:
                        query += ", "
                query += f") VALUES (ST_SetSRID(ST_MakePoint({lon},{lat}), 4326), "
                for count, type in enumerate(rand_var_types):
                    if type != "int":
                        query += "'"
                    query += f'{row[1][count+1]}'
                    if type != "int":
                        query += "'"
                    if count < len(rand_var_names)-1:
                        query += ", "
                query += '); \r'           

            else:
                full_var_names = rand_var_names + extra_var_name
                full_var_types = rand_var_types + extra_var_types

                extra_values = []
                for i in range(len(extra_var_name)):
                    extra_values.append(row[1][len(rand_var_names)+i])

                query = f"INSERT into {table_name} (thegeom, "
                for count, type in enumerate(full_var_names):
                    query += f"{full_var_names[count]}"
                    if count < len(full_var_names)-1:
                        query += ", "
                query += f") VALUES (ST_SetSRID(ST_MakePoint({lon},{lat}), 4326), "
                for count, type in enumerate(full_var_types):
                    if type != "int":
                        query += "'"
                    
                    if type == "str":
                        query += "{}".format(row[1][count+1].replace("'", "''")) 
                    else:
                        query += "{}".format(row[1][count+1])
                    
                    if type != "int":
                        query += "'"
                    if count < len(full_var_names)-1:
                        query += ", "
                query += '); \r'

        else:
            if(not extra_var):
                # Insert statement for each point along with the included variables
                query = "INSERT into {} (thegeom) VALUES (ST_SetSRID(ST_MakePoint({},{}),4326)); \r".format(
                    table_name, lon, lat)
            else:
                extra_values = []
                for i in range(len(extra_var_name)):
                    extra_values.append(row[1][i+1])

                var_index = 1
                query = f"INSERT into {table_name} (thegeom, "
                for count, type in enumerate(extra_var_name):
                    query += f"{extra_var_name[count]}"
                    if count < len(extra_var_name)-1:
                        query += ", "
                query += f") VALUES (ST_SetSRID(ST_MakePoint({lon},{lat}), 4326), "
                for count, type in enumerate(extra_var_types):
                    if type != "int":
                        query += "'"
                    
                    if type == "str":
                        query += "{}".format(row[1][count+1].replace("'", "''")) 
                    else:
                        query += "{}".format(row[1][count+1])
                    
                    if type != "int":
                        query += "'"
                    if count < len(extra_var_name)-1:
                        query += ", "
                query += '); \r'

        # Write query string to SQL file
        sqlFile.write(query)

    print("Successfully printed {} rows to {} with table name: {}.".format(num_rows, f'{directory}/SQL/{table_name}.sql', table_name))

def csv_distribute(filename, num_values):
    source = pd.read_csv(filename)
    if source.shape[1] < 2:
        return list(random.choices(source['string'], k = num_values))
    return list(random.choices(source['string'], weights = source['weight'], k = num_values))

def generate_vars(gdf, rand_var_types, rand_var_names, rand_var_params):
    for count, type in enumerate(rand_var_types):
        if type == 'int':
            gdf[f'{rand_var_names[count]}'] = [randint(rand_var_params[count][0], rand_var_params[count][1]) for i in range(len(gdf.index))]
        elif type == 'str':
            gdf[f'{rand_var_names[count]}'] = [''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(rand_var_params[count])) for i in range(len(gdf.index))]
        elif type == 'ts':
            start = pd.to_datetime(rand_var_params[count][0])
            end = pd.to_datetime(rand_var_params[count][1])
            ts_start = start.value//10**9
            ts_end = end.value//10**9
            gdf[f'{rand_var_names[count]}'] = list(pd.to_datetime(np.random.randint(ts_start, ts_end, len(gdf.index)), unit='s'))
        else:
            return

    print("Random variables generated...")
    return gdf

########### PRIMARY & SECONDARY GENERATION ##########

def points_ratio(total_pts, ratio):
    bulk_points = round(total_pts * ratio)
    local_points = total_pts - bulk_points
    return bulk_points, local_points

def primary_generation(source, source_centroid, total_pts, rand_centroid, epsg):
    
    if(rand_centroid):
        vor_polygons = voronoi_gen(source, source_centroid, 256, 'rand')
    else:
        vor_polygons = voronoi_gen(source, source_centroid, 256, 'eq')

    vor_union = gpd.GeoDataFrame(vor_polygons.dissolve(by='class', as_index=False)).set_crs(3857)
    
    if(total_pts > 0):
        vor_all = []
        for i in range(len(vor_union['geometry'])):
            current = cascaded_union(list(vor_union['geometry'][0:i + 1]))
            vor_all.append(current)

        vor_all = gpd.GeoDataFrame(pd.DataFrame(vor_all, columns=['geometry']), geometry='geometry', crs=3857)

        vor_points = int(np.ceil(total_pts / 5))
        print("Primary points per region", vor_points)
        primary_pts = gpd.GeoDataFrame(pd.DataFrame([], columns=['geometry']), geometry='geometry', crs=3857)
        for i in range(len(vor_all)):
            #if total_pts % 5 != 0:
            #    if i == len(vor_all.iloc[0])-1:
            #        temp = vor_points * i
            #        vor_points = bulk_points - temp
            if(rand_centroid):
                gdf = points_moving_centre(vor_all['geometry'][i], vor_points)
            else:
                gdf = points_centre(vor_all['geometry'][i], vor_points)

            primary_pts = pd.concat([primary_pts, gdf], ignore_index=True)
    else:
        primary_pts = []

    #vor_pts = gpd.GeoDataFrame(pd.DataFrame(primary_pts, columns=['geometry']), geometry='geometry', crs=source.crs)
    
    return primary_pts.iloc[0:total_pts].to_crs(epsg), vor_union.to_crs(epsg)

def secondary_gen_equal(source, source_centroid, total_pts, vor_num, epsg):
    print("Starting secondary generation with equal-area Voronoi...")
    if vor_num > 256:
        vor_num = 256
        print("Max vor_num is 256!")
    elif vor_num <= 0:
        vor_num = 1
        print("Min vor_num is 1!")

    local_vor_points = int(np.ceil(total_pts / vor_num))

    local_vor_polygons = voronoi_gen(source, source_centroid, vor_num, 'eq')

    # the polyogn crs is set as the main polygon crs
    #local_vor_polygons.crs = source.crs

    #local_gdf = gpd.GeoDataFrame()
    local_gdf = gpd.GeoDataFrame(pd.DataFrame([], columns=['geometry']), geometry='geometry', crs=epsg)
    for i in range(0, vor_num):
        #if local_points % vor_num != 0:
        #    if i == vor_num-1:
        #        temp = local_vor_points * i
        #        local_vor_points = local_points - temp

        current = points_centre(local_vor_polygons['geometry'][i], local_vor_points)
        #local_gdf = gpd.GeoDataFrame(local_gdf.append(current, ignore_index=True))
        local_gdf = pd.concat([local_gdf, current], ignore_index=True)

    return local_gdf, local_vor_polygons

def secondary_gen_var_area(source, source_centroid, total_pts, vor_num, epsg):
    print("Starting secondary generation with variable-area Voronoi and area-based points...")
    if vor_num > 128:
        vor_num = 128
        print("Max poly_area value is 128!")
    elif vor_num <= 0:
        vor_num = 1
        print("Min vor_num is 1!")

    local_vor_points = round(total_pts / vor_num)

    print("\tGenerating secondary Voronoi regions...")
    local_vor_polygons = voronoi_gen(source, source_centroid, vor_num, 'area')

    # calculating the area of each polygon to determine the proportion of points in each
    local_area_union = local_vor_polygons.dissolve()
    local_area = local_area_union.area

    local_gdf = gpd.GeoDataFrame(pd.DataFrame([], columns=['geometry']), geometry='geometry', crs=epsg)

    print("\tBeginning Secondary points generation...")
    for i in range(0, vor_num):
        #if i == vor_num-1:
        #    current_local_points = int(local_points - len(local_gdf))
        #else:
        area_prop = local_vor_polygons['geometry'][i].area / local_area
        current_local_points = int(total_pts * area_prop)
        current = points_centre(local_vor_polygons['geometry'][i], current_local_points)
        local_gdf = pd.concat([local_gdf, current], ignore_index=True)
    print("\tSecondary generation complete.")
    return local_gdf, local_vor_polygons

def secondary_gen_var_equal(source, source_centroid, total_pts, vor_num, epsg):
    print("Starting secondary generation with variable-area Voronoi and equal points...")
    if vor_num > 128:
        vor_num = 128
        print("Max poly_area value is 128!")
    elif vor_num <= 0:
        vor_num = 1
        print("Min vor_num is 1!")

    local_vor_points = round(total_pts / vor_num)

    print("\tGenerating Secondary Voronoi regions...")
    local_vor_polygons = voronoi_gen(source, source_centroid, vor_num, 'area')

    # the polyogn crs is set as the main polygon crs
    #local_vor_polygons.crs = source.crs

    local_gdf = gpd.GeoDataFrame(pd.DataFrame([], columns=['geometry']), geometry='geometry', crs=epsg)
    print("\tBeginning Secondary points generation...")
    for i in range(0, vor_num):
        #if local_points % vor_num != 0:
        #    if i == vor_num-1:
        #        temp = local_vor_points * i
        #        local_vor_points = local_points - temp
        current = points_centre(local_vor_polygons['geometry'][i], local_vor_points)
        local_gdf = pd.concat([local_gdf, current], ignore_index=True)

    print("\tSecondary generation complete.")
    return local_gdf, local_vor_polygons

def secondary_generation(source, source_centroid, total_pts, gen_type, vor_num, epsg):
    #### Local level generation ###

    # gen_type:
        # 0 for no local-level generation
        # 1 for Equal-area Voronoi local generation
        # 2 for Variable-area Voronoi local generation with points determined by area
        # 3 for Variable-area Voronoi local generation with equal points in each Voronoi

    print("*"*65)

    # Set no local generation as the default
    if gen_type > 3:
        gen_type = 0


    # Local generation with approximately equal-area Voronoi polygons
    if gen_type == 1:
        local_gdf, local_vor_polygons = secondary_gen_equal(source, source_centroid, total_pts, vor_num, 3857)
        
    # Local generation with variable area Voronoi polygons with number of points based on area
    elif gen_type == 2:
        local_gdf, local_vor_polygons = secondary_gen_var_area(source, source_centroid, total_pts, vor_num, 3857)

    # Local generation with variable area Voronoi polygons with equal number of points in each
    elif gen_type == 3:
        local_gdf, local_vor_polygons = secondary_gen_var_equal(source, source_centroid, total_pts, vor_num, 3857)
    else:
        print("Skipping secondary generation.")
        local_gdf = gpd.GeoDataFrame([])

    #local_gdf = gpd.GeoDataFrame(pd.DataFrame(loca, columns=['geometry']), geometry='geometry', crs=source.crs)

    return local_gdf.iloc[0:total_pts].reset_index(drop=True).to_crs(epsg), local_vor_polygons.reset_index(drop=True).to_crs(epsg)

########## OUTPUT FUNCTIONS

def plot_output(polygon, polygon_centroid, buffers, voronoi, vor_centroid, primary_points, secondary_points, basemap, epsg):
    fig, axs = plt.subplots(2,2, figsize=(10,8))
    fig.tight_layout()
    fig.suptitle(f"RADIAN Synthetic Spatial Data Generator\n{len(primary_points)} primary points, {len(secondary_points)} secondary points\nProjected to EPSG:{epsg}")

    subtitles = ["Primary Generation", "Secondary Generation", "Full Generation", "Full Generation (basemap)"]

    # Primary generation
    buffers.plot(ax=axs[0,0], cmap='Blues', edgecolor='white')
    polygon.centroid.plot(ax=axs[0,0], color='red', markersize=3)
    primary_points.plot(ax=axs[0,0], color='green', markersize=1.5)

    # Secondary generation
    voronoi.plot(ax=axs[0,1], cmap='Blues', edgecolor='white')
    #vor_centroid.plot(ax=axs[0,1], color='red', markersize=3)
    secondary_points.plot(ax=axs[0,1], color='green', markersize=1.5)
    vor_centroid.plot(ax=axs[0,1], color='red', markersize=10)

    # Full generation
    buffers.plot(ax=axs[1,0], cmap='Blues', edgecolor='white', alpha=0.25)
    voronoi.plot(ax=axs[1,0], cmap='Blues', edgecolor='white', alpha=0.25)

    for ax in axs.flatten()[2:4]:
        primary_points.plot(ax=ax, color='green', markersize=1.5)
        secondary_points.plot(ax=ax, color='green', markersize=1.5)

    # Printing source polygon border and setting subplot titles
    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")
        ax.set_title(subtitles[i],y=0.05, pad=-14)
        polygon.plot(ax=ax, edgecolor='black', facecolor='none')
        polygon_centroid.plot(ax=ax, color='red', markersize=10)

    # Basemap
    if basemap: cx.add_basemap(axs[1,1], attribution=False)
    plt.show()

def radian():
    global global_accepted_points
    global global_rejected_points
    global global_ratio_list

    start_time = time.time()
    params = json.load(open("parameters.json"))
    set_seed = params["set_seed"]
    directory = os.path.dirname(params["filename"])

    filename = params["filename"]
    epsg = params['epsg']
    save_name = os.path.basename(filename)
    total_pts = params["total_pts"]
    gen_type = params["gen_type"]
    ratio = (params["ratio"] / 100)
    vor_num = params["vor_num"]
    table_name = params["table_name"]
    random_vars = params["random_vars"]
    rand_var_types = params["rand_var_types"]
    rand_var_names = params["rand_var_names"]
    rand_var_params = params["rand_var_params"]
    rand_centroid = params["rand_centroid"]
    to_sql = params["to_sql"]
    to_geojson = params["to_geojson"]
    vor_to_geojson = params["vor_to_geojson"]
    plot = params["plot"]
    basemap = params["basemap"]
    preview = params["preview"]
    extra_var = params["extra_var"]
    extra_var_types = params["extra_var_types"]
    extra_var_name = params["extra_var_name"]
    extra_var_file = params["extra_var_file"]
    extra_var_params = params["extra_var_params"]
    extra_var_dict = params["extra_var_dict"]


    global glob_random_seed

    if set_seed:
        glob_random_seed = params["seed"]
        random.seed(glob_random_seed)
    else:
        glob_random_seed = random.randint(0, 2147483647)
        random.seed(glob_random_seed)

    # Global diagnostic variables

    # Reading in the GeoJSON file and setting the CRS to a meter-based projection
    print("Reading {}...".format(filename))
    source_gdf = gpd.read_file(filename)
    source_gdf = source_gdf.to_crs(epsg=3857)
    print("\t{} loaded as source polygon.".format(filename))

    source_area = source_gdf.to_crs(epsg=8858)
    print("\tPolygon area = " + str(round(source_gdf.area[0], 2)) + "m^2")
    dens = total_pts/source_area.area[0]
    print("\tPoints density is " + str(dens) + " per m^2")
    if(dens > 1):
        print("Points density too low: {} points in an area of {}".format(total_pts, source.area))
        print("Minimum points density is 1 point / m^2.")
        return

    ########## POINTS GENERATION ##########

    source = source_gdf.loc[0, 'geometry']
    
    if(rand_centroid):
        source_centroid = moving_centroid(source,epsg)
    else:
        source_centroid = source.centroid

    primary_total, secondary_total = points_ratio(total_pts, ratio)

    print("Generating", primary_total, "primary points and ", secondary_total, "secondary points")

    print("Beginning Primary generation...")
    primary_points, primary_vor_polygons = primary_generation(source, source_centroid, primary_total, rand_centroid, epsg)
    
    print("CRS values:")
    print("Primary points:", primary_points.crs)
    print("Primary voronoi:", primary_vor_polygons.crs)

    
    print("\tPrimary generation complete.")

    print("Beginning Secondary generation...")
    secondary_points, local_vor_polygons = secondary_generation(source, source_centroid, secondary_total, gen_type, vor_num, epsg)
    print("\tSecondary generation complete.")
    
    print("CRS values:")
    print("Secondary points:", secondary_points.crs)
    print("Secondary voronoi:", local_vor_polygons.crs)



    # Merging the bulk and local point dataframes for output to SQL or GeoJSON

    gdf_out = gpd.GeoDataFrame(primary_points.append(secondary_points, ignore_index=True), crs=epsg)


    ########## ADDITIONAL METADATA GENERATION ##########

    if random_vars:
        gdf_out = generate_vars(gdf_out, rand_var_types, rand_var_names, rand_var_params)

    if(extra_var):
        print("Generating metadata...")
        for variable in extra_var_dict:
            gdf_out[f'{variable["name"]}'] = csv_distribute(variable['source'], total_pts)

        print("Metadata generated.")
        print("*" * 60)


    ########## EXPORTING OF DATA ##########

    # Set exported CRS
    gdf_out = gdf_out.to_crs(epsg)
    print("gdf.crs = ", gdf_out.crs)

    # Exporting data to SQL dump file
    if(to_sql):
        print("Exporting to SQL...")
        if not os.path.exists(f"{directory}/SQL"):
            os.makedirs(f"{directory}/SQL")
        gdf_to_sql(table_name, gdf_out, total_pts, random_vars, rand_var_types, rand_var_names, rand_var_params, extra_var, extra_var_types, extra_var_name, directory)
        print("*" * 60)

    # Exporting  data to GeoJSON
    if(to_geojson):
        print("Exporting to GeoJSON...")
        if not os.path.exists(f"{directory}/GeoJSON"):
            os.makedirs(f"{directory}/GeoJSON")
        gdf_out.insert(0, 'PKID', range(0, len(gdf_out)))
        gdf_out.to_file(f"{directory}/GeoJSON/{table_name}.geojson", driver='GeoJSON')
        print("\tSuccessfully created GeoJSON file {}.geojson with {} points".format(table_name, total_pts))
        print("*" * 60)

    # Exporting voronoi polygons
    if(gen_type > 0 and vor_to_geojson):
        # To GeoJSON
        print("Exporting Voronoi polygons to GeoJSON...")
        if not os.path.exists(f"{directory}/GeoJSON"):
            os.makedirs(f"{directory}/GeoJSON")
        #local_vor_polygons.insert(0, 'PKID', range(0, len(local_vor_polygons)))
        local_vor_polygons['class'] = local_vor_polygons['class'].astype(int)
        local_vor_polygons.to_file(f"{directory}/GeoJSON/{table_name}_voronoi_polygons.geojson", driver='GeoJSON')
        print("\tSuccessfully created GeoJSON file {}_voronoi_polygons.geojson".format(table_name))
        print("*" * 60)
        print(local_vor_polygons)

        print("Exporting Voronoi Polygons to SQL")
        if not os.path.exists(f"{directory}/SQL"):
            os.makedirs(f"{directory}/SQL")

        # To SQL
        gdf_poly_to_sql("voronoi_poly_test", local_vor_polygons, directory)

    ########## PRINTING GENERATION DIAGNOSTICS ##########
    """
    print("Generation info:")
    print("\tTotal accepted points: " + str(global_accepted_points))
    print("\tTotal rejected points: " + str(global_rejected_points))
    print("\tRejection ratio: " + str(global_accepted_points / global_rejected_points))
    glob_ratio_list.append(global_accepted_points / global_rejected_points)

    diag_text = str("Rejection ratio list: " + str(glob_ratio_list) + "\n")
    diag_text += "Mean ratio: " + str(sum(glob_ratio_list)/len(glob_ratio_list))

    print("\tGeneration seed: " + str(glob_random_seed))
    """
    end_time = time.time()

    print("Generation time taken = ", (end_time-start_time))

    ########### DATA PREVIEW ##########

    if(preview):
        print(f"Total points: {len(gdf_out)}\n", gdf_out.head())

    ########## PLOTTING DATA ##########

    vor_centroids = gpd.GeoDataFrame(pd.DataFrame(local_vor_polygons.centroid, columns=['geometry']), geometry='geometry', crs=epsg)
    print(vor_centroids)

    if(plot):
        #print(source.crs, source_centroid.to_crs(epsg).crs, primary_vor_polygons.crs, local_vor_polygons.crs, vor_centroids.crs, primary_points.crs, secondary_points.crs)
        plot_output(source_gdf.to_crs(epsg), source_centroid.to_crs(epsg), primary_vor_polygons, local_vor_polygons, vor_centroids, primary_points, secondary_points, False, epsg)

def radian_uniform():

    start_time = time.time()
    params = json.load(open("parameters.json"))
    set_seed = params["set_seed"]

    filename = params["filename"]
    epsg = params['epsg']
    total_pts = params["total_pts"]

    # Global diagnostic variables

    # Reading in the GeoJSON file and setting the CRS to a meter-based projection
    print("Reading {}...".format(filename))
    source_gdf = gpd.read_file(filename)
    source_gdf = source_gdf.to_crs(epsg=3857)
    print("\t{} loaded as source polygon.".format(filename))

    ########## POINTS GENERATION ##########

    source = source_gdf.loc[0, 'geometry']
    gdf_out = points_uniform(source, total_pts, 3857)
    
        ########## EXPORTING OF DATA ##########

    # Set exported CRS
    gdf_out = gdf_out.to_crs(epsg)
    
    end_time = time.time()

    print("Generation time taken = ", (end_time-start_time))

    ########### DATA PREVIEW ##########

    print(f"Total points: {len(gdf_out)}\n", gdf_out.head())

radian()