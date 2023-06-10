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
import matplotlib.gridspec as gridspec
import contextily as cx
import warnings
from tqdm import tqdm

# voroni generation packages
from shapely.ops import unary_union, cascaded_union
from geovoronoi import voronoi_regions_from_coords, points_to_coords

# k-means clustering packages
from sklearn.cluster import KMeans

# misc json and shapely packages
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
    if ratio < 1.8:
        return 1.8
    return ratio

########## POINT GENERATION ##########

def points_uniform(poly, num_points):
    global global_rejected_points
    global global_accepted_points
    min_x, min_y, max_x, max_y = poly.bounds
    poly_ratio = poly_bb_ratio(poly)
    poly_gdf = gpd.GeoDataFrame(pd.DataFrame([poly], columns=['geometry']), geometry='geometry', crs=3857)

    points = []
    # Generates points repeatedly with a uniform generation within the bounds of the polygon
    pbar = tqdm(desc=f'Generating points...', total=num_points)
    while len(points) < round(num_points * poly_ratio):
        points.append(Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)]))
        pbar.update(1)
    gdf = gpd.GeoDataFrame(pd.DataFrame(points, columns=['geometry']), geometry='geometry', crs=3857)
    gdf = gdf.sjoin(poly_gdf, predicate='within')
    gdf = gdf.drop(['index_right'], axis=1)

    old_length = len(gdf)
    global_rejected_points += old_length - num_points
    pbar.close()
    return gdf.iloc[0:num_points].reset_index(drop=True)
    
def points_moving_centre(poly, num_points):
    global global_rejected_points
    global global_accepted_points
    min_x, min_y, max_x, max_y = poly.bounds
    cx, cy = poly.centroid.x, poly.centroid.y
    max_pt = Point(max_x, max_y)
    radius = max_pt.distance(poly.centroid)

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


    old_length = len(points_gdf)
    global_rejected_points += old_length - num_points

    # Points list is converted to a GeoDataFrame and outputted
    return points_gdf.iloc[0:num_points].reset_index(drop=True)

def points_centre(poly, num_points):
    global global_rejected_points
    global global_accepted_points
    min_x, min_y, max_x, max_y = poly.bounds
    # When a multipolygon is generated and passed here an error is thrown - need to catch seed of it not working
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

    old_length = len(points_gdf)
    global_rejected_points += old_length - num_points

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
        gdf_centroid = poly_centroid
        vor_centroids = kmeans_centroids(poly, 500, vor_num, 0)
    # Setting crs to meter based projection

    # Convert the boundary geometry into a union of the polygon
    #boundary_shape = cascaded_union(poly) Depreciated
    boundary_shape = unary_union(poly)
    coords = points_to_coords(vor_centroids.geometry)

    # Calculating the voronoi regions
    region_polys, region_pts = voronoi_regions_from_coords(coords, boundary_shape)


    df = pd.DataFrame(list(region_polys.items()), columns=['index','geometry'])
    gdf_poly = gpd.GeoDataFrame(df, geometry='geometry', crs=3857)

    # Calculating distance of Voronoi polygons to the centroid (moving or original)
    gdf_poly['dist_to_centre'] = 0
    for i in range(vor_num):
        if(gen_type == 'rand'):
            current = gdf_poly['geometry'][i].centroid.distance(gdf_centroid.iloc[0].geometry)
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

    return gdf_poly.drop('index', axis=1)

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

    for row in gdf.itertuples():
        poly_coords = row[1].wkt
        query = f"INSERT into {table_name} (thegeom, "
        query += f"dist_to_centre, poly_class"
        query += f") VALUES (ST_SetSRID(ST_PolygonFromText('{poly_coords}'),3857), "
        query += f"{row[2]}, {row[3]}); \r"
        sqlFile.write(query)
    
    return


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

    if extra_var:
        for variable in extra_var_dict:
            create_query = f",\n\t{variable['name']}"
            if variable['type'] == 'str':
                create_query += 'VARCHAR'
            else:
                create_query += 'INTEGER'
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

    return gdf

########### PRIMARY & SECONDARY GENERATION ##########

def points_ratio(total_pts, ratio):
    bulk_points = round(total_pts * ratio)
    local_points = total_pts - bulk_points
    return bulk_points, local_points

def primary_generation(source, source_centroid, total_pts, rand_centroid, epsg):
    global global_rejected_points

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
        primary_pts = gpd.GeoDataFrame(pd.DataFrame([], columns=['geometry']), geometry='geometry', crs=3857)
        for i in range(len(vor_all)):
            if(rand_centroid):
                gdf = points_moving_centre(vor_all['geometry'][i], vor_points)
            else:
                gdf = points_centre(vor_all['geometry'][i], vor_points)

            primary_pts = pd.concat([primary_pts, gdf], ignore_index=True)
    else:
        primary_pts = []
    
    old_length = len(primary_pts)
    global_rejected_points += old_length - total_pts

    missing_pts = total_pts - len(primary_pts)
    while(missing_pts > 0):
        temp_points = points_uniform(source, missing_pts*2)
        primary_pts = pd.concat([temp_points, primary_pts], ignore_index=True)
        missing_pts = total_pts - len(primary_pts)

    return primary_pts.iloc[0:total_pts].to_crs(epsg), vor_union.to_crs(epsg)

def secondary_gen_equal(source, source_centroid, total_pts, vor_num, epsg):
    #print("Starting secondary generation with equal-area Voronoi...")
    if vor_num > 256:
        vor_num = 256
        print("Max vor_num is 256!")
    elif vor_num <= 0:
        vor_num = 1
        print("Min vor_num is 1!")

    local_vor_points = int(np.ceil(total_pts / vor_num))

    local_vor_polygons = voronoi_gen(source, source_centroid, vor_num, 'eq')

    local_gdf = gpd.GeoDataFrame(pd.DataFrame([], columns=['geometry']), geometry='geometry', crs=epsg)
    for i in range(0, vor_num):
        current = points_centre(local_vor_polygons['geometry'][i], local_vor_points)
        local_gdf = pd.concat([local_gdf, current], ignore_index=True)

    return local_gdf.reset_index(drop=True), local_vor_polygons

def secondary_gen_var_area(source, source_centroid, total_pts, vor_num, epsg):
    #print("Starting secondary generation with variable-area Voronoi and area-based points...")
    if vor_num > 128:
        vor_num = 128
        print("Max poly_area value is 128!")
    elif vor_num <= 0:
        vor_num = 1
        print("Min vor_num is 1!")

    local_vor_points = round(total_pts / vor_num)

    local_vor_polygons = voronoi_gen(source, source_centroid, vor_num, 'area')

    # calculating the area of each polygon to determine the proportion of points in each
    local_area_union = local_vor_polygons.dissolve()
    local_area = local_area_union.area

    local_gdf = gpd.GeoDataFrame(pd.DataFrame([], columns=['geometry']), geometry='geometry', crs=epsg)

    for i in range(0, vor_num):
        area_prop = local_vor_polygons['geometry'][i].area / local_area
        current_local_points = int(total_pts * area_prop)
        current = points_centre(local_vor_polygons['geometry'][i], current_local_points)
        local_gdf = pd.concat([local_gdf, current], ignore_index=True)
    return local_gdf.reset_index(drop=True), local_vor_polygons

def secondary_gen_var_equal(source, source_centroid, total_pts, vor_num, epsg):
    #print("Starting secondary generation with variable-area Voronoi and equal points...")
    if vor_num > 128:
        vor_num = 128
        print("Max poly_area value is 128!")
    elif vor_num <= 0:
        vor_num = 1
        print("Min vor_num is 1!")

    local_vor_points = round(total_pts / vor_num)

    local_vor_polygons = voronoi_gen(source, source_centroid, vor_num, 'area')

    # the polyogn crs is set as the main polygon crs
    local_gdf = gpd.GeoDataFrame(pd.DataFrame([], columns=['geometry']), geometry='geometry', crs=epsg)
    for i in range(0, vor_num):
        current = points_centre(local_vor_polygons['geometry'][i], local_vor_points)
        local_gdf = pd.concat([local_gdf, current], ignore_index=True)

    return local_gdf.reset_index(drop=True), local_vor_polygons

def secondary_generation(source, source_centroid, total_pts, gen_type, vor_num, epsg):
    #### Local level generation ###
    global global_rejected_points

    # gen_type:
        # 0 for no local-level generation
        # 1 for Equal-area Voronoi local generation
        # 2 for Variable-area Voronoi local generation with points determined by area
        # 3 for Variable-area Voronoi local generation with equal points in each Voronoi

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
        local_gdf = gpd.GeoDataFrame([])

    old_length = len(local_gdf)
    global_rejected_points += old_length - total_pts

    missing_pts = total_pts - len(local_gdf)
    if missing_pts > 0:
        temp_points = points_uniform(source, missing_pts)
        local_gdf = pd.concat([temp_points, local_gdf], ignore_index=True)

    return local_gdf.iloc[0:total_pts].reset_index(drop=True).to_crs(epsg), local_vor_polygons.reset_index(drop=True).to_crs(epsg)

########## OUTPUT FUNCTIONS

def plot_output(polygon, polygon_centroid, buffers, voronoi, vor_centroid, primary_points, secondary_points, basemap, epsg):
    minx, miny, maxx, maxy = polygon.total_bounds
    width = maxx - minx
    height = maxy - miny

    aspect_ratio = width/height
    fig_width = 8
    fig_height = 8

    if aspect_ratio > 1:
        fig_height = fig_width / round(aspect_ratio)
    else:
        fig_width = fig_height / round(aspect_ratio)

    if basemap:
        fig, axs = plt.subplots(2,2, figsize=(fig_width, fig_height))
        axs = axs.flatten()
    else:
        fig = plt.figure(figsize=(fig_width,fig_height))
        gs = gridspec.GridSpec(4, 4)
        ax1 = plt.subplot(gs[0:2, :2])
        ax2 = plt.subplot(gs[0:2, 2:4])
        ax3 = plt.subplot(gs[2:4, 1:3])
        axs = [ax1, ax2, ax3]
    
    fig.tight_layout()
    fig.suptitle(f"{len(primary_points)} primary points, {len(secondary_points)} secondary points - Projected to EPSG:{epsg}")
    plt.get_current_fig_manager().set_window_title("RADIAN (\u03C0) - Synthetic Spatial Data Generator")

    subtitles = ["Primary Generation", "Secondary Generation", "Full Generation", "Full Generation (basemap)"]
    point_size = 0.5

    # Primary generation
    buffers.plot(ax=axs[0], cmap='Blues', edgecolor='white')
    polygon.centroid.plot(ax=axs[0], color='red', markersize=3)
    primary_points.plot(ax=axs[0], color='green', markersize=point_size)

    # Secondary generation
    voronoi.plot(ax=axs[1], cmap='Blues', edgecolor='white')
    #vor_centroid.plot(ax=axs[0,1], color='red', markersize=3)
    secondary_points.plot(ax=axs[1], color='green', markersize=point_size)
    vor_centroid.plot(ax=axs[1], color='red', markersize=10)

    # Full generation
    buffers.plot(ax=axs[2], cmap='Blues', edgecolor='white', alpha=0.25)
    voronoi.plot(ax=axs[2], cmap='Blues', edgecolor='white', alpha=0.25)

    for ax in axs[2:4]:
        primary_points.plot(ax=ax, color='green', markersize=point_size)
        secondary_points.plot(ax=ax, color='green', markersize=point_size)

    # Printing source polygon border and setting subplot titles
    for i, ax in enumerate(axs):
        ax.axis("off")
        ax.set_title(subtitles[i],y=0.05, pad=-14)
        polygon.plot(ax=ax, edgecolor='black', facecolor='none')
        polygon_centroid.plot(ax=ax, color='red', markersize=10)

    # Basemap
    if basemap: cx.add_basemap(axs[3], attribution=False)
    plt.show()

def radian():
    global global_accepted_points
    global global_rejected_points
    global global_ratio_list

    print("RADIAN (\u03C0) - Synthetic Spatial Data Generator")

    star_width = 128
    print('*' * star_width)

    # Loading running parameters from 'parameters.json'
    start_time = time.time()
    params = json.load(open("parameters.json"))
    set_seed = params["set_seed"]
    directory = os.path.dirname(params["filepath"])
    filepath = params["filepath"]
    epsg = params['epsg']
    save_name = os.path.basename(filepath)
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
    vor_to_sql = params["vor_to_sql"]
    plot = params["plot"]
    basemap = params["basemap"]
    preview = params["preview"]
    extra_var = params["extra_var"]
    extra_var_types = params["extra_var_types"]
    extra_var_name = params["extra_var_name"]
    extra_var_file = params["extra_var_file"]
    extra_var_params = params["extra_var_params"]
    extra_var_dict = params["extra_var_dict"]

    # Setting generation seed
    global glob_random_seed
    if set_seed:
        glob_random_seed = params["seed"]
        random.seed(glob_random_seed)
    else:
        glob_random_seed = random.randint(0, 2147483647)
        random.seed(glob_random_seed)

    print("* Generation seed: " + str(glob_random_seed))

    # Reading in the GeoJSON file, projecting to EPSG:3857 and checking for points density
    print(f"* Reading {filepath}...")
    file_name = os.path.basename(filepath)
    _, file_extension = os.path.splitext(file_name)
    if file_extension != '.geojson':
        print("invalid polygon file - please use .geojson format")
        return
    source_gdf = gpd.read_file(filepath)
    source_gdf = source_gdf.to_crs(epsg=3857)
    source_area = source_gdf.to_crs(epsg=8858)
    dens = total_pts/source_area.area[0]
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

    print(f"* Generating {primary_total} primary points and {secondary_total} secondary points in {file_name}")

    print(f"* Primary generation using {'moving centroid' if rand_centroid else 'true centroid'}")
    primary_points, primary_vor_polygons = primary_generation(source, source_centroid, primary_total, rand_centroid, epsg)

    if gen_type == 0:
        print(f"* No secondary generation.")
    print(f"* Secondary generation with {'equal area voronoi regions' if gen_type == 1 else ('variable-area voronoi regions by area' if gen_type == 2 else 'variable-area voronoi with equal points')}")
    secondary_points, local_vor_polygons = secondary_generation(source, source_centroid, secondary_total, gen_type, vor_num, epsg)


    #print(f"* Actual generation: {len(primary_points)} primary points, {len(secondary_points)} secondary points")

    # Merging the bulk and local point dataframes for output to SQL or GeoJSON

    gdf_out = gpd.GeoDataFrame(primary_points.append(secondary_points, ignore_index=True), crs=epsg)     

    print("* Points generated successfully!")
    print('*' * star_width)

    ########## ADDITIONAL METADATA GENERATION ##########

    if random_vars or extra_var:
        print("Generating Metadata:")

    if random_vars:
        gdf_out = generate_vars(gdf_out, rand_var_types, rand_var_names, rand_var_params)

    if(extra_var):
        #print("Generating metadata...")
        for variable in extra_var_dict:
            gdf_out[f'{variable["name"]}'] = csv_distribute(variable['source'], total_pts)


    ########## EXPORTING OF DATA ##########

    # Set exported CRS
    gdf_out = gdf_out.to_crs(epsg)
    source_gdf = source_gdf.to_crs(epsg) 

    # Exporting data to SQL dump file
    if(to_sql):
        if not os.path.exists(f"{directory}/SQL"):
            os.makedirs(f"{directory}/SQL")
        #def gdf_to_sql(table_name, gdf, num_rows, random_vars, rand_var_types, rand_var_names, extra_var, extra_var_types, extra_var_name, extra_var_dict, directory):
        gdf_to_sql(table_name, gdf_out, total_pts, random_vars, rand_var_types, rand_var_names, extra_var, extra_var_types, extra_var_name, extra_var_dict, directory)
        print("* SQL dump file created: {} rows to {} with table name: {}.".format(total_pts, f'{directory}/SQL/{table_name}.sql', table_name))
    
    # Exporting  data to GeoJSON
    if(to_geojson):
        if not os.path.exists(f"{directory}/GeoJSON"):
            os.makedirs(f"{directory}/GeoJSON")
        gdf_out.insert(0, 'PKID', range(0, len(gdf_out)))
        gdf_out.to_file(f"{directory}/GeoJSON/{table_name}.geojson", driver='GeoJSON')
        print("* Successfully created GeoJSON file {}.geojson with {} points".format(table_name, total_pts))

    # Exporting voronoi polygons
    if(gen_type > 0 and vor_to_geojson):
        # To GeoJSON
        if not os.path.exists(f"{directory}/GeoJSON"):
            os.makedirs(f"{directory}/GeoJSON")
        # make copy of local vor polys to set class to correct type and then export that?
        local_vor_polygons['class'] = local_vor_polygons['class'].astype(int)
        local_vor_polygons.to_file(f"{directory}/GeoJSON/{table_name}_voronoi_polygons.geojson", driver='GeoJSON')
        print("* Successfully created GeoJSON file {}_voronoi_polygons.geojson".format(table_name))

    if(gen_type > 0 and vor_to_sql):
        if not os.path.exists(f"{directory}/SQL"):
            os.makedirs(f"{directory}/SQL")
        # To SQL
        gdf_poly_to_sql("voronoi_poly_test", local_vor_polygons, directory)
        print("* SQL dump file created: {} rows to {} with table name: voronoi_poly_test.".format(total_pts, f'{directory}/SQL/{table_name}.sql'))

    end_time = time.time()

    ########## PRINTING GENERATION DIAGNOSTICS ##########
    
    print('*' * star_width)

    print("Generation Information:")
    print("* Total rejected points: " + str(global_rejected_points))
    print("* Rejection ratio: " + str(total_pts / global_rejected_points))
    
    print("* Generation time taken = ", (end_time-start_time))

    ########### DATA PREVIEW ##########

    print('*' * star_width)

    if(preview):
        print(f"Data Preview: Total points: {len(gdf_out)}\n", gdf_out.head())

    ########## PLOTTING DATA ##########()

    if(plot):
        plot_output(source_gdf, source_gdf.centroid, primary_vor_polygons, local_vor_polygons, local_vor_polygons.centroid, primary_points, secondary_points, False, epsg)

    print('*' * star_width)

def radian_uniform(total_pts, source):

    start_time = time.time()

    
    ########## POINTS GENERATION ##########

    gdf_out = points_uniform(source.loc[0, 'geometry'], total_pts)
    
    ########## EXPORTING OF DATA ##########

    end_time = time.time()
    
    time_taken = end_time - start_time

    return time_taken 


def uniform_benchmark(filepath, iterations):
    print("Reading {}...".format(filepath))
    # Reading in the GeoJSON file and setting the CRS to a meter-based projection
    source_gdf = gpd.read_file(filepath)
    source_gdf = source_gdf.to_crs(epsg=3857)
    save_dir = os.path.dirname(filepath)
    if not os.path.exists(f"{save_dir}/QGIS_compare"):
        os.makedirs(f"{save_dir}/QGIS_compare")
    
    points_list = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    output = pd.DataFrame([])
    print(f"* Generating points for {iterations} iterations...")
    for x in range(iterations):
        print("** iteration", x)
        gen_times = []
        for points in points_list:
            print(f"* {x}: {points} points...")
            gen_start = time.time()
            gdf_out = points_uniform(source_gdf.loc[0, 'geometry'], points)
            print(f"{len(gdf_out)} points generated.")
            gdf_out.insert(0, 'PKID', range(0, len(gdf_out)))
            gen_end = time.time()

            gen_times.append(gen_end - gen_start)

        df = pd.DataFrame(zip([x+1 for i in range(len(points_list))], points_list, gen_times), columns=['experiment','points','gen_time'])
        output = pd.concat([output, df])
        print("done.")
    print("* Iterations complete.")
    output = output.reset_index(drop=True)

    print("Saving data to .csv...")
    data_path = f"{os.path.dirname(filepath)}/QGIS"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    output.to_csv(f"{data_path}/radian_uniform_{iterations}.csv", index=False)

    print(output)

def qgis_compare(filepath, iterations=1):
    radian_data_path = f"{os.path.dirname(filepath)}/QGIS/radian_uniform_{iterations}.csv"
    if not os.path.exists(radian_data_path):
        print(f"* Running RADIAN uniform generation for {iterations} iterations...")
        uniform_benchmark(filepath, iterations)
    else:
        print("* File already exists - skipping generation...")
    print("Loading RADIAN data...")
    radian_data = pd.read_csv(radian_data_path)
    
    print("Plotting data...")
    groups = radian_data.groupby('experiment')
    for name, group in groups:
        plt.plot(group.points, group.time, marker='o', linestyle='', markersize=5, label=name)

    plt.show()

uniform_benchmark('scenarios/usa/usa.geojson', 1)

#radian()