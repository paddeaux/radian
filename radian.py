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

def voronoi_gen(poly, vor_num, gen_type):
    poly = poly.to_crs(epsg=3857)
    # Voronoi centroids are generated based on the specified generation type
    if(gen_type == 'eq'): # Equal-area uniformly distributed Voronoi regions
        vor_centroids = kmeans_centroids(poly, 500, vor_num, 0)
    elif(gen_type == 'area'): # Variable area, centrally focused Voronoi regions
        vor_centroids = kmeans_centroids(poly, 500, vor_num, 1)
    elif(gen_type == 'rand'): # Equal-area uniformly distributed Voronoi regions (with moving centroid)
        # Calculate moving centroid in an eliptical region around the original centroid
        boundary_poly = poly['geometry'][0]
        cx, cy = boundary_poly.centroid.x, boundary_poly.centroid.y
        min_x, min_y, max_x, max_y = boundary_poly.bounds

        #max_pt = Point(max_x, max_y)
        #radius = max_pt.distance(boundary_poly.centroid)
        range_x = (max_x - min_x) / 4
        range_y = (max_y - min_y) / 16

        # Generation of the moving centroid
        centroid_point = Point([random.uniform(cx - range_x, cx + range_x), random.uniform(cy - range_y, cy + range_y)])
        gdf_centroid = gpd.GeoSeries(centroid_point)
        gdf_centroid.crs = poly.crs

        vor_centroids = kmeans_centroids(poly, 500, vor_num, 0)
    # Setting crs to meter based projection
    gdf_proj = vor_centroids.set_crs(poly.crs)

    # Convert the boundary geometry into a union of the polygon
    boundary_shape = cascaded_union(poly.geometry)
    coords = points_to_coords(gdf_proj.geometry)

    # Calculating the voronoi regions
    region_polys, region_pts = voronoi_regions_from_coords(coords, boundary_shape)

    # Create GeoDataFrame of the Voronoi Polygons

    ####### This is the 'Polygon' object is not iterable error source ##########################
    #df = pd.DataFrame.from_dict(region_polys, orient='index', columns=['geometry'])

    df = pd.DataFrame(list(region_polys.items()), columns=['index','geometry'])
    gdf_poly = gpd.GeoDataFrame(df, geometry='geometry')
    gdf_poly.crs = poly.crs

    # Calculating distance of Voronoi polygons to the centroid (moving or original)
    gdf_poly['dist_to_centre'] = " "
    for i in range(vor_num):
        if(gen_type == 'rand'):
            current = gdf_poly['geometry'][i].centroid.distance(gdf_centroid[0])
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
            centroid = gdf_centroid[0]
            c_current = centroid.buffer(dist_break * (i + 1))
            buffers.append(c_current)

    circ_df = pd.DataFrame(buffers, columns=['geometry'])
    circ_gdf = gpd.GeoDataFrame(circ_df, geometry='geometry')

    vor_union = gdf_poly.dissolve(by='class', as_index=False)


    #fig, ax = plt.subplots(1,3,figsize=(16,8))
    #gdf_poly.plot(ax=ax[1], edgecolor='white', facecolor='grey')
    #gdf_poly.plot(ax=ax[2], cmap='Blues')

    #vor_centroids.plot(ax=ax[1], color='green', markersize=0.8)
    #for x in ax:
    #    poly.plot(ax=x, facecolor='none', edgecolor='red')
    #    vor_centroids.plot(ax=x, color='green', markersize=0.8)
    #plt.show()

    return gdf_poly

# This function takes in a Shapely Polygon, number of points, number of clusters, and a generation type, and returns
# a GeoDataFrame of Points, representing centroids produced by a Kmeans clustering of uniformly random points generated
# inside of the polygon. The centroids will either be generated using uniform generation, with points distributed
# uniformly in the polygon, or with points concentrated towards the polygon centroid.

def kmeans_centroids(poly, num_points, num_cluster, eq_area):
    # Points are generated randomly in the polygon
    if(eq_area == 0): # Uniform distribution
        source = random_point_gen(poly['geometry'][0], num_points, 'uni')
    else: # Centroid-focused distribution
        source = random_point_gen(poly['geometry'][0], num_points, 'cent')

    source = source.set_crs(epsg=3857)

    # The geometries of the Shapely points are converted to a numpy array for use in the kmeans algorithm
    feature_coords = np.array([[e.x, e.y] for e in source.geometry])

    # A kmeans object is created using the specified number of clusters
    kmeans = KMeans(num_cluster, random_state=glob_random_seed)
    kmeans.fit(feature_coords)

    # The cluster centres are stored as centroids, and this list is put into a GeoDataFrame and returned
    centroids = kmeans.cluster_centers_
    df = pd.DataFrame(centroids, columns=['x', 'y'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    gdf.crs = source.crs

    fig, ax = plt.subplots(figsize=(10,5))

    return gdf

# This function takes in a GeoDataFrame of randomly generated points (along with additional random variables)
# and produces a SQL file that will allow a PostgreSQL table to be created containing the data

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

def gdf_to_sql(table_name, gdf, num_rows, default_vars, rand_var_types, rand_var_names, rand_var_params, extra_var, extra_var_types, extra_var_name, png_filename, directory):
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

    if default_vars:
        sqlFile.write(',\n')
        for count, type in enumerate(rand_var_types):
            sqlFile.write(f'\t{rand_var_names[count]} {get_var_type(rand_var_types[count])}')
            if count < len(rand_var_types)-1:
                sqlFile.write(', \n')
        #sqlFile.write(',\n\t{} {}, \n'.format(rand_var_names[0], get_var_type(rand_var_types[0])))
        #sqlFile.write('\t{} {}, \n'.format(rand_var_names[1], get_var_type(rand_var_types[1])))
        #sqlFile.write('\t{} {}'.format(rand_var_names[2], get_var_type(rand_var_types[2])))

    if extra_var:
        for var_name in extra_var_name:
            create_query = ',\n\t{} '.format(var_name)
            if isinstance(gdf[f'{var_name}'][0], np.int64):
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
        if default_vars:
            if (not extra_var):
                var_index = 1
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

            
            # indexing for extra variables begins at position 1
            #rand_str = row[1][1]
            #rand_int = row[1][2]
            #rand_ts = row[1][3]
            #if(not extra_var):
                # Insert statement for each point along with the included variables
            #    query = f"INSERT into {table_name} (thegeom, "
            #    query += f"{rand_var_names[0]}, {rand_var_names[1]}, {rand_var_names[2]}"
            #    query += f") VALUES (ST_SetSRID(ST_MakePoint({lon},{lat}),4326), "
            #    query += f"{rand_int}, '{rand_str}', '{rand_ts}'); \r"
            

            else:
                full_var_names = rand_var_names + extra_var_name
                full_var_types = rand_var_types + extra_var_types

                extra_values = []
                for i in range(len(extra_var_name)):
                    extra_values.append(row[1][len(rand_var_names)+i])

                var_index = 1
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

# This function takes in a Shapely Polygon, number of points to be generated, and generation type, and produces
# a set of random points within that polygon, generated in a pattern set by the generation type:
# 'uni' = Points are uniformly distributed within the polygon, meaning there is no clustering or other patterns
# 'rand' = Points are generated around a randomly generated "moving centroid", with points concentrated around the centroid
# 'cent' = Points are generated around the polygon centroid, with points centred around it

def random_point_gen(poly, num_points, gen_type, return_buffers=False):
    min_x, min_y, max_x, max_y = poly.bounds
    cx, cy = poly.centroid.x, poly.centroid.y
    max_pt = Point(max_x, max_y)
    radius = max_pt.distance(poly.centroid)

    global global_rejected_points
    global global_accepted_points

    # Make list to hold all points and variables
    points = []
    rand_strings = []
    rand_ints = []
    rand_ts = []
    buffers = []

    # Uniform generation
    if (gen_type == 'uni'):
        # Generates points repeatedly with a uniform generation within the bounds of the polygon
        while len(points) < num_points:
            random_point = Point(
                [random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
            # Once points lie within the polygon, they are appended to the list
            if (random_point.within(poly)):
                points.append(random_point)

        # List of points is converted to a GeoDataFrame
        df = pd.DataFrame(points, columns=['geometry'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        return gdf

    # Moving centroid generation
    elif (gen_type == 'rand'):
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
        # Points are generated section by section, along with randomly generated int and string variables
        for i in range(0, section_num):
            if num_points % 5 != 0:
                if i == 4:
                    temp = section_pts * i
                    section_pts = num_points - temp
            # Current circular buffer is created within which to generate points
            point_current = Point([centroid_point.x + (cent_diff_x * i), centroid_point.y + (cent_diff_y * i)])
            c_current = point_current.buffer(section_size * (i + 1))

            # This while loop controls the generation of points in the current section
            current_pts = 0
            while current_pts < section_pts:
                # here we generate a point using a uniform distribution to set the possible x and y ranges
                random_point = Point([random.uniform(min_x + random.uniform(0,100), max_x - random.uniform(0,100)), random.uniform(min_y + random.uniform(0,100), max_y - random.uniform(0,100))])

                # Here the random point is added to the list if it lies inside the bounds of both the polygon
                # as well as within the current section
                if (random_point.within(poly)):
                    if (random_point.within(c_current)):
                        points.append(random_point)
                        current_pts += 1
                        global_accepted_points += 1
                    else:
                        global_rejected_points += 1
                else:
                    global_rejected_points += 1
                if (current_pts >= section_pts):
                    section_num += 1

            buffers.append(c_current)

        # GeoDataFrame of the circular buffers is created (optional use for visualization)
        df_buff = pd.DataFrame(buffers, columns=['geometry'])
        gdf_buff = gpd.GeoDataFrame(df_buff, geometry='geometry')

        # Points list is converted to a GeoDataFrame and outputted
        df = pd.DataFrame(points, columns=['geometry'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        # gdf.crs = poly.crs
        return gdf

    # Original centroid generation
    elif (gen_type == 'cent'):

        section_num = 5
        section_size = (radius * 0.8) / section_num
        section_pts = round(num_points / section_num)
        for i in range(0, section_num):
            if num_points % 5 != 0:
                if i == 4:
                    temp = section_pts * i
                    section_pts = num_points - temp
            c_current = poly.centroid.buffer(section_size * (i+1))
            current_pts = 0
            while current_pts < section_pts:
                random_point = Point([random.uniform(min_x + random.uniform(0,100), max_x - random.uniform(0,100)), random.uniform(min_y + random.uniform(0,100), max_y - random.uniform(0,100))])
                if (random_point.within(poly)):
                    if (random_point.within(c_current)):
                        points.append(random_point)
                        current_pts += 1
                        global_accepted_points += 1
                    else:
                        global_rejected_points += 1
                else:
                    global_rejected_points += 1
                if (current_pts >= section_pts):
                    section_num += 1
            buffers.append(c_current)
        df_buff = pd.DataFrame(buffers, columns=['geometry'])
        gdf_buff = gpd.GeoDataFrame(df_buff, geometry='geometry')

        df = pd.DataFrame(points, columns=['geometry'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        return gdf

def csv_att(filename, num_values):
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

def get_default_parameters():
    default_param_dict = {
        "filename":"scenarios/Kildare/maynooth.geojson",
        "total_pts":500,
        "gen_type":2,
        "ratio":50,
        "vor_num": 16,
        "table_name":"CS621SQLTesting",
        "default_vars" : False,
        "rand_var_types" : ["ts", "int", "ts"],
        "rand_var_names" : ["timestamp_one", "integer_one", "timestamp_three"],
        "rand_var_params" : [["2022-01-01 00:00:00", "2022-12-31 23:59:59"],[1,999],["2022-01-01 00:00:00", "2022-12-31 23:59:59"]],
        "rand_centroid": True,
        "int_range": [1,1000],
        "string_len": 15,
        "timestamp_range": ["2022-01-01 00:00:00", "2022-12-31 23:59:59"],
        "to_sql": True,
        "to_geojson":True,
        "to_png": True,
        "vor_to_geojson": True,
        "vor_to_sql": True,
        "png_filename":"output_default.png",
        "plot": True,
        "breakdown": False,
        "basemap": True,
        "extra_var": False,
        "extra_var_types" : ["str", "str"],
        "extra_var_name":["var_string", "string_equal_dist"],
        "extra_var_file":["restaurant.csv", "restaurant_no_weights.csv"],
        "set_seed": False,
        "seed" : 1878019349
    }

    return default_param_dict

# Radial points generation using JSON file for parameters
def radial_spatial_points(png_filename, directory):
    global global_accepted_points
    global global_rejected_points
    global global_ratio_list

    filename = params["filename"]
    save_name = os.path.basename(filename)
    total_pts = params["total_pts"]
    gen_type = params["gen_type"]
    ratio = (params["ratio"] / 100)
    vor_num = params["vor_num"]
    table_name = params["table_name"]
    default_vars = params["default_vars"]
    rand_var_types = params["rand_var_types"]
    rand_var_names = params["rand_var_names"]
    rand_var_params = params["rand_var_params"]
    rand_centroid = params["rand_centroid"]
    to_sql = params["to_sql"]
    to_geojson = params["to_geojson"]
    to_png = params["to_png"]
    vor_to_geojson = params["vor_to_geojson"]
    png_filename = params["png_filename"]
    plot = params["plot"]
    basemap = params["basemap"]
    breakdown = params["breakdown"]
    extra_var = params["extra_var"]
    extra_var_types = params["extra_var_types"]
    extra_var_name = params["extra_var_name"]
    extra_var_file = params["extra_var_file"]

    # Global diagnostic variables

# Reading in the GeoJSON file and setting the CRS to a meter-based projection
    print("Reading {}...".format(filename))
    source = gpd.read_file(filename)
    source = source.to_crs(epsg=3857)
    print("\t{} loaded as source polygon.".format(filename))

    source_area = source.to_crs(epsg=8858)
    print("\tPolygon area = " + str(round(source.area[0], 2)) + "m^2")
    dens = total_pts/source_area.area[0]
    print("\tPoints density is " + str(dens) + " per m^2")
    if(dens > 1):
        print("Points density too low: {} points in an area of {}".format(total_pts, source.area))
        print("Minimum points density is 1 point / m^2.")
        return
    min_x, min_y, max_x, max_y = source.total_bounds

    # Set the number used for Voronoi-based buffer generation to 256
    print("*"*65)
    # This set of if-else statements determines how the Voronoi polygons are generated
    if(params['rand_centroid']):
        print("Generating Voronoi with Moving Centroid...")
        vor_polygons = voronoi_gen(source, 256, 'rand')
    else:
        print("Generating Voronoi with Original Centroid...")
        vor_polygons = voronoi_gen(source, 256, 'eq')

    vor_polygons.crs = source.crs
        # Source-boundary Generation

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
        print("Starting secondary generation with equal-area Voronoi...")
        if vor_num > 256:
            vor_num = 256
            print("Max vor_num is 256!")
        elif vor_num <= 0:
            vor_num = 1
            print("Min vor_num is 1!")

        local_points = round(total_pts * (1 - ratio)) # 1 - ratio
        local_vor_points = int(local_points / vor_num)

        local_vor_polygons = voronoi_gen(source, vor_num, 'eq')

        # the polyogn crs is set as the main polygon crs
        local_vor_polygons.crs = source.crs

        local_gdf = gpd.GeoDataFrame()
        for i in range(0, vor_num):
            if local_points % vor_num != 0:
                if i == vor_num-1:
                    temp = local_vor_points * i
                    local_vor_points = local_points - temp

            current = random_point_gen(local_vor_polygons['geometry'][i], local_vor_points, 'cent')
            local_gdf = gpd.GeoDataFrame(local_gdf.append(current, ignore_index=True ))
        print("\tSecondary generation complete.")

    # Local generation with variable area Voronoi polygons with number of points based on area
    elif gen_type == 2:
        print("Starting secondary generation with variable-area Voronoi and area-based points...")
        if vor_num > 128:
            vor_num = 128
            print("Max poly_area value is 128!")
        elif vor_num <= 0:
            vor_num = 1
            print("Min vor_num is 1!")

        local_points = round(total_pts * (1 - ratio))
        local_vor_points = round(local_points / vor_num)

        print("\tGenerating secondary Voronoi regions...")
        local_vor_polygons = voronoi_gen(source, vor_num, 'area')

        # calculating the area of each polygon to determine the proportion of points in each
        local_area_union = local_vor_polygons.dissolve()
        local_area = local_area_union.area

        local_gdf = gpd.GeoDataFrame()

        print("\tBeginning Secondary points generation...")
        for i in range(0, vor_num):
            if i == vor_num-1:
                current_local_points = int(local_points - len(local_gdf))
            else:
                area_prop = local_vor_polygons['geometry'][i].area / local_area
                current_local_points = int(local_points * area_prop)
            current = random_point_gen(local_vor_polygons['geometry'][i], current_local_points, 'cent')
            local_gdf = gpd.GeoDataFrame(local_gdf.append(current, ignore_index=True))
        print("\tSecondary generation complete.")

    # Local generation with variable area Voronoi polygons with equal number of points in each
    elif gen_type == 3:
        print("Starting secondary generation with variable-area Voronoi and equal points...")
        if vor_num > 128:
            vor_num = 128
            print("Max poly_area value is 128!")
        elif vor_num <= 0:
            vor_num = 1
            print("Min vor_num is 1!")

        local_points = round(total_pts * (1 - ratio))
        local_vor_points = round(local_points / vor_num)

        print("\tGenerating Secondary Voronoi regions...")
        local_vor_polygons = voronoi_gen(source, vor_num, 'area')

        # the polyogn crs is set as the main polygon crs
        local_vor_polygons.crs = source.crs

        local_gdf = gpd.GeoDataFrame()
        print("\tBeginning Secondary points generation...")
        for i in range(0, vor_num):
            if local_points % vor_num != 0:
                if i == vor_num-1:
                    temp = local_vor_points * i
                    local_vor_points = local_points - temp
            current = random_point_gen(local_vor_polygons['geometry'][i], local_vor_points, 'cent')
            local_gdf = gpd.GeoDataFrame(local_gdf.append(current, ignore_index=True))

        print("\tSecondary generation complete.")

    else:
        print("Skipping secondary generation.")

    print("*"*65)

    # Bulk generation in the source polygon
    vor_union = vor_polygons.dissolve(by='class', as_index=False)

    bulk_points = round(total_pts * ratio)
    if(bulk_points > 0):
        print("Beginning Primary generation...")
        # dissolve the vor_polygons by the class, determined by their distance to the centre of the polygon
        # combining successive Voronoi regions to allow points generating inside them
        vor_all = gpd.GeoDataFrame()
        print("\tCreating Voronoi-based buffers...")
        for i in range(len(vor_union['geometry'])):
            current = gpd.GeoSeries(cascaded_union(list(vor_union['geometry'][0:i + 1])))
            vor_all = gpd.GeoDataFrame(vor_all.append(current, ignore_index=True))

        

        # Generating points inside the merged Voronoi regions
        print("\tBeginning Primary points generation...")
        vor_points = round(bulk_points / 5)
        vor_pts = gpd.GeoDataFrame()
        for i in range(len(vor_all[0])):
            ### should this be the rand???
            if bulk_points % 5 != 0:
                if i == len(vor_all[0])-1:
                    temp = vor_points * i
                    vor_points = bulk_points - temp
            if(rand_centroid):
                gdf = random_point_gen(vor_all[0][i], vor_points, 'rand')
            else:
                gdf = random_point_gen(vor_all[0][i], vor_points, 'cent')

            vor_pts = gpd.GeoDataFrame(vor_pts.append(gdf, ignore_index=True))

        print("\tPrimary generation complete.")
    else:
        print("Skipping Primary generation.")

    print("*"*65)

    # Merging the bulk and local point dataframes for output to SQL or GeoJSON
    if(to_sql or to_geojson or extra_var or plot):
        print("Generating random variable data...")
        if(gen_type != 0 and bulk_points > 0):
            gdf_out = gpd.GeoDataFrame(vor_pts.append(local_gdf, ignore_index=True))
        elif(gen_type == 0):
            gdf_out = vor_pts
        else:
            gdf_out = local_gdf

        gdf_out = gdf_out.set_crs(epsg=3857)
        gdf_out = gdf_out.to_crs(epsg=4326)

        if default_vars:
            gdf_out = generate_vars(gdf_out, rand_var_types, rand_var_names, rand_var_params)

            """
            # Here we can set the extra random attribute variables
            gdf_out['rand_str'] = [''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(string_len)) for i in range(total_pts)]
            gdf_out['rand_int'] = [randint(int_range[0], int_range[1]) for i in range(total_pts)]

            def random_dates(start, end, n):
                ts_start = start.value//10**9
                ts_end = end.value//10**9
                return list(pd.to_datetime(np.random.randint(ts_start, ts_end, n), unit='s'))

            start = pd.to_datetime(timestamp_range[0])
            end = pd.to_datetime(timestamp_range[1])
            gdf_out['rand_ts'] = random_dates(start, end, total_pts)

            gdf_out = gdf_out.rename(columns={"rand_int": f"{rand_var_names[0]}", "rand_str": f"{rand_var_names[1]}", "rand_ts": f"{rand_var_names[2]}"})
            print("\tDefault random variables generated.")
            print("*" * 60)
            """

    # Plotting of generated points and Voronoi regions
    if(plot or to_png):
        # create fig and axes to plot and compare points and Voronoi buffer regions

        # Setting plot title
        title = "Random Spatial Data Generation - Seed: {}\n" \
                "{} total points at ratio of  {}:{} -> Primary Points = {}, Secondary Points = {}\n".format(glob_random_seed, total_pts, int(ratio*100),int((1-ratio)*100), bulk_points, total_pts-bulk_points)
        if (rand_centroid):
            title += "Using moving centroid for primary generation\n"
        else:
            title += "Using original centroid for primary generation\n"

        # Local Generation type
        if (gen_type == 0):
            title += "Secondary Generation: None\n"
        elif (gen_type == 1):
            title += "Secondary Generation: Equal-area Voronoi, {} local regions\n".format(vor_num)
        elif (gen_type == 2):
            title += "Secondary Generation: Variable-area Voronoi - Points by Area, {} local regions\n".format(
                vor_num)
        elif (gen_type == 3):
            title += "Secondary Generation: Variable-area Voronoi - Equal Points, {} local regions\n".format(vor_num)
        else:
            title += "Secondary Generation: Error"

        if(breakdown):
            fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(16, 5))
            fig.tight_layout()

            source.plot(ax=ax1, color='gray')
            source.plot(ax=ax2, color='gray')
            source.plot(ax=ax3, color='gray')

            vor_union.plot(ax=ax1, cmap='Blues', edgecolor='white', alpha=0.6)
            if gen_type != 0:
                local_vor_polygons.plot(ax=ax2, cmap='Blues', edgecolor='white', alpha=0.4)

            # vor_polygons.plot(ax=ax1, column='class', cmap='Blues', edgecolor='white', alpha=0.8)
            # centroid_point.plot(ax=ax1, color='Red')

            # plot the Bulk points
            print_points = True

            if print_points:
                if(bulk_points > 0):
                    vor_pts.plot(ax=ax1, markersize=0.4, color='black')
                    vor_pts.plot(ax=ax3, markersize=0.4, color='black')

                if gen_type != 0:
                    local_gdf.plot(ax=ax2, markersize=0.4, color='white')
                    local_gdf.plot(ax=ax3, markersize=0.4, color='black')
            
            #fig.suptitle(title)  # Plot title text
            #ax1.set_title("Primary Generation",y=0.05, pad=-14)
            ax1.axis("off")
            #ax2.set_title("Secondary Generation",y=0.05, pad=-14)
            ax2.axis("off")
            #ax3.set_title("Final Points generation",y=0.05, pad=-14)
            ax3.axis("off")

            #plt.axis('equal')

        else:
            fig, ax = plt.subplots(figsize=(6,6))
            source = source.to_crs(epsg=3857)
            gdf_plot = gdf_out.to_crs(epsg=3857)

            if basemap:
                source.plot(ax=ax, facecolor="none", edgecolor='red')
                gdf_plot.plot(ax=ax, markersize=1, color='red', edgecolor='black')
                cx.add_basemap(ax, attribution=False)
            else:
                source.plot(ax=ax, color='gray')
                gdf_plot.plot(ax=ax, markersize=1, color='red')

            title = "Seed: " + str(glob_random_seed)
            #fig.suptitle(title)  # Plot title text
            ax.set_title(title)
            ax.axis("off")
            plt.axis('equal')

        if(to_png):
            print("Exporting to PNG...")
            plt.savefig(f"{directory}/PNG/{save_name.split('.')[0]}_points_{png_filename}.png")
            print("*" * 60)

        if(plot):
            print("Plotting output...")
            plt.show()
            print("*" * 60)

    if(extra_var):
        print("Generating extra attribute data...")
        if(isinstance(extra_var_file, str)):
            gdf_out[f'{extra_var_name}'] = csv_att(extra_var_file, total_pts)
            print("Extra attribute generated.")
        elif(isinstance(extra_var_file, list)):
            for i in range(len(extra_var_name)):
                gdf_out[f'{extra_var_name[i]}'] = csv_att(extra_var_file[i], total_pts)
            print("Extra attributes generated.")
        else:
            print("Extra var input invalid")

        print("*" * 60)

    # Exporting the generated points to an SQL file
    if(to_sql):
        print("Exporting to SQL...")
        if not os.path.exists(f"{directory}/SQL"):
            os.makedirs(f"{directory}/SQL")
        gdf_to_sql(table_name, gdf_out, total_pts, default_vars, rand_var_types, rand_var_names, rand_var_params, extra_var, extra_var_types, extra_var_name, png_filename, directory)
        print("*" * 60)

    # Exporting the generated points to a GeoJSON file
    if(to_geojson):
        print("Exporting to GeoJSON...")
        if not os.path.exists(f"{directory}/GeoJSON"):
            os.makedirs(f"{directory}/GeoJSON")
        gdf_out.insert(0, 'PKID', range(0, len(gdf_out)))
        gdf_out.to_file(f"{directory}/GeoJSON/{table_name}.geojson", driver='GeoJSON')
        print("\tSuccessfully created GeoJSON file {}.geojson with {} points".format(table_name, total_pts))
        print("*" * 60)

    if(gen_type > 0 and vor_to_geojson):
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

        gdf_poly_to_sql("voronoi_poly_test", local_vor_polygons, directory)


    #global glob_ratio_list
    print("Generation info:")
    print("\tTotal accepted points: " + str(global_accepted_points))
    print("\tTotal rejected points: " + str(global_rejected_points))
    print("\tRejection ratio: " + str(global_accepted_points / global_rejected_points))
    glob_ratio_list.append(global_accepted_points / global_rejected_points)


start_time = time.time()

glob_ratio_list = []

default_run = False

if default_run:
    params = get_default_parameters()
else:
    params = json.load(open("parameters.json"))

set_seed = params["set_seed"]
directory = os.path.dirname(params["filename"])

if set_seed:
    glob_random_seed = params["seed"]
    random.seed(glob_random_seed)
else:
    glob_random_seed = random.randint(0, 2147483647)
    random.seed(glob_random_seed)

for run in range(0,1):
    radial_spatial_points(png_filename=f"{run}", directory=directory)

diag_text = str("Rejection ratio list: " + str(glob_ratio_list) + "\n")
diag_text += "Mean ratio: " + str(sum(glob_ratio_list)/len(glob_ratio_list))

f = open("rejection.txt", "w")
f.write(diag_text)
f.close()

print("\tGeneration seed: " + str(glob_random_seed))

end_time = time.time()

print("Generation time taken = ", (end_time-start_time))