# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:54:29 2022

@author: paddy
"""
import random
from random import randint

import json
import string
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx

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

    print("region_polys:\n" + str(region_polys))

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

    return gdf_poly

# This fucntion takes in a Shapely Polygon, number of points, number of clusters, and a generation type, and returns
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
    kmeans = KMeans(num_cluster)
    kmeans.fit(feature_coords)

    # The cluster centres are stored as centroids, and this list is put into a GeoDataFrame and returned
    centroids = kmeans.cluster_centers_
    df = pd.DataFrame(centroids, columns=['x', 'y'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    gdf.crs = source.crs
    return gdf

# This function takes in a GeoDataFrame of randomly generated points (along with additional random variables)
# and produces a SQL file that will allow a PostgreSQL table to be created containing the data

def gdf_to_sql(table_name, gdf, num_rows, extra_var, extra_var_name):
    # Opens up an SQL file based on the table name, writes to the file and closes it
    sqlFile = open(f'{table_name}_SQL.sql', "w")
    sqlFile.write("")
    sqlFile.close()

    # Opens up the SQL file to append lines to it
    sqlFile = open(f'{table_name}_SQL.sql', "a")

    # SQL statments to create the table as well as drop if exists the table are appended
    sqlFile.write('DROP TABLE IF EXISTS {}; \n'.format(table_name))
    sqlFile.write('CREATE TABLE {} ( \n'.format(table_name))
    sqlFile.write('pkid SERIAL PRIMARY KEY NOT NULL, \n')
    sqlFile.write("thegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326), \n")
    sqlFile.write('rand_int INTEGER, \n')
    sqlFile.write('rand_string VARCHAR, \n')
    sqlFile.write('rand_ts TIMESTAMP')
    if extra_var:
        for var_name in extra_var_name:
            create_query = ',\n{} '.format(var_name)
            if isinstance(gdf[f'{var_name}'][0], np.int64):
                create_query += 'INTEGER'
            else:
                create_query += 'VARCHAR'
            sqlFile.write(create_query)

    sqlFile.write('\n); \n')

    # Creation of Spatial Index for the SQL file
    sqlFile.write('CREATE INDEX {}_spatial_index ON {} USING gist (thegeom); \n'.format(table_name, table_name))

    # Loop through every point in the GeoDataFrame to write an instert statement to append to the SQL file
    for row in gdf.iterrows():
        # Set 'y' value to Latitude and 'x' value to Longitude.
        lat = row[1][0].y
        lon = row[1][0].x
        # Pull the randomly generated strings and ints from the dataframe
        rand_str = row[1][1]
        rand_int = row[1][2]
        rand_ts = row[1][3]
        if(not extra_var):
            # Insert statement for each point along with the included variables
            query = "INSERT into {} (thegeom, rand_int, rand_string, rand_ts) VALUES (ST_SetSRID(ST_MakePoint({},{}),4326), {}, '{}', '{}'); \r".format(
                table_name, lon, lat, rand_int, rand_str, rand_ts)
        else:
            extra_values = []
            for i in range(len(extra_var_name)):
                extra_values.append(row[1][4+i])

            query = 'INSERT into {} (thegeom, rand_int, rand_string, rand_ts'.format(table_name)
            for var_name in extra_var_name:
                query += ', {}'.format(var_name)

            query += " ) VALUES (ST_SetSRID(ST_MakePoint({},{}),4326), {}, '{}', '{}'".format(
                lon, lat, rand_int, rand_str, rand_ts)

            for value in extra_values:
                if isinstance(value, str):
                    temp = value.replace("'", "''")
                    query += ", '{}'".format(temp)
                elif isinstance(value, int):
                    query += ', {}'.format(value)

            query += '); \r'

        # Write query string to SQL file
        sqlFile.write(query)

    print("Successfully printed {} rows to {} with table name: {}.".format(num_rows, f'{table_name}_SQL.sql', table_name))

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
                if (current_pts >= section_pts):
                    section_num += 1

            buffers.append(c_current)

        # GeoDataFrame of the circular buffers is created (optional use for visualization)
        df_buff = pd.DataFrame(buffers, columns=['geometry'])
        gdf_buff = gpd.GeoDataFrame(df_buff, geometry='geometry')

        # Points list is converted to a GeoDataFrame and outputted
        df = pd.DataFrame(points, columns=['geometry'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf.crs = poly._crs
        return gdf

    # Original centroid generation
    elif (gen_type == 'cent'):

        section_num = 5
        section_size = (radius * 0.8) / section_num
        section_pts = round(num_points / section_num)
        for i in range(0, section_num):
            if num_points % 5 != 0:
                print("num_points % 5 is true")
                if i == 4:
                    print("i == section_num-1 is true")
                    temp = section_pts * i
                    section_pts = num_points - temp
            print("Section pts is : " + str(section_pts))
            c_current = poly.centroid.buffer(section_size * (i+1))
            current_pts = 0
            while current_pts < section_pts:
                random_point = Point([random.uniform(min_x + random.uniform(0,100), max_x - random.uniform(0,100)), random.uniform(min_y + random.uniform(0,100), max_y - random.uniform(0,100))])
                if (random_point.within(poly)):
                    if (random_point.within(c_current)):
                        points.append(random_point)
                        current_pts += 1
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
    name_dist = list(random.choices(source['string'], weights = source['weight'], k = num_values))
    return name_dist

# Radial points generation using JSON file for parameters
def radial_spatial_points():

    params = json.load(open('parameters.json'))
    filename = params["filename"]
    total_pts = params["total_pts"]
    gen_type = params["gen_type"]
    ratio = params["ratio"]
    vor_num = params["vor_num"]
    rand_centroid = params["rand_centroid"]
    int_range = params["int_range"]
    string_len = params["string_len"]
    timestamp_range = params["timestamp_range"]
    to_sql = params["to_sql"]
    to_geojson = params["to_geojson"]
    to_png = params["to_png"]
    png_filename = params["png_filename"]
    plot = params["plot"]
    basemap = params["basemap"]
    breakdown = params["breakdown"]
    extra_var = params["extra_var"]
    extra_var_name = params["extra_var_name"]
    extra_var_file = params["extra_var_file"]

# Reading in the GeoJSON file and setting the CRS to a meter-based projection
    source = gpd.read_file(params['filename'])
    source = source.to_crs(epsg=3857)
    dens = total_pts/source.area[0]
    if(dens > 1):
        print("Points density too low: {} points in an area of {}".format(total_pts, source.area))
        print("Minimum points density is 1 point / m^2.")
        return
    min_x, min_y, max_x, max_y = source.total_bounds

    # Set the number used for Voronoi-based buffer generation to 256

    # This set of if-else statements determines how the Voronoi polygons are generated
    if(params['rand_centroid']):
        print("Generating Voronoi with Moving Centroid...")
        vor_polygons = voronoi_gen(source, 256, 'rand')
    else:
        print("Generating Voronoi with Original...")
        vor_polygons = voronoi_gen(source, 256, 'eq')

    vor_polygons.crs = source.crs
        # Source-boundary Generation

    #### Local level generation ###

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
        if vor_num > 256:
            vor_num = 256
            print("Max vor_num is 256!")
        elif vor_num <= 0:
            vor_num = 1
            print("Min vor_num is 1!")

        local_points = round(total_pts * (1 - ratio))
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

    # Local generation with variable area Voronoi polygons with number of points based on area
    elif gen_type == 2:
        if vor_num > 128:
            vor_num = 128
            print("Max poly_area value is 128!")
        elif vor_num <= 0:
            vor_num = 1
            print("Min vor_num is 1!")

        local_points = round(total_pts * (1 - ratio))
        local_vor_points = round(local_points / vor_num)

        local_vor_polygons = voronoi_gen(source, vor_num, 'area')

        # calculating the area of each polygon to determine the proportion of points in each
        local_area_union = local_vor_polygons.dissolve()
        local_area = local_area_union.area

        local_gdf = gpd.GeoDataFrame()

        for i in range(0, vor_num):
            if i == vor_num-1:
                current_local_points = int(local_points - len(local_gdf))
            else:
                area_prop = local_vor_polygons['geometry'][i].area / local_area
                current_local_points = int(round(local_points * area_prop))
            print("current local points is :" + str(current_local_points))
            current = random_point_gen(local_vor_polygons['geometry'][i], current_local_points, 'cent')
            local_gdf = gpd.GeoDataFrame(local_gdf.append(current, ignore_index=True))

    # Local generation with variable area Voronoi polygons with equal number of points in each
    elif gen_type == 3:
        if vor_num > 128:
            vor_num = 128
            print("Max poly_area value is 128!")
        elif vor_num <= 0:
            vor_num = 1
            print("Min vor_num is 1!")

        local_points = round(total_pts * (1 - ratio))
        local_vor_points = round(local_points / vor_num)

        print("Generating Voronoi with Variable Area...")
        local_vor_polygons = voronoi_gen(source, vor_num, 'area')

        print("Voronoi complete.")
        # the polyogn crs is set as the main polygon crs
        local_vor_polygons.crs = source.crs

        local_gdf = gpd.GeoDataFrame()
        for i in range(0, vor_num):
            if local_points % vor_num != 0:
                if i == vor_num-1:
                    temp = local_vor_points * i
                    local_vor_points = local_points - temp
            current = random_point_gen(local_vor_polygons['geometry'][i], local_vor_points, 'cent')
            local_gdf = gpd.GeoDataFrame(local_gdf.append(current, ignore_index=True))

    # Printing out examples of the generated Voronoi regions


    # Bulk generation in the source polygon
    vor_union = vor_polygons.dissolve(by='class', as_index=False)

    bulk_points = round(total_pts * ratio)
    if(bulk_points > 0):
        # dissolve the vor_polygons by the class, determined by their distance to the centre of the polygon
        # combining successive Voronoi regions to allow points generating inside them
        vor_all = gpd.GeoDataFrame()
        for i in range(len(vor_union['geometry'])):
            current = gpd.GeoSeries(cascaded_union(list(vor_union['geometry'][0:i + 1])))
            vor_all = gpd.GeoDataFrame(vor_all.append(current, ignore_index=True))

        # Generating points inside the merged Voronoi regions
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

    # Merging the bulk and local point dataframes for output to SQL or GeoJSON
    if(to_sql or to_geojson or extra_var):
        if(gen_type != 0 and bulk_points > 0):
            gdf_out = gpd.GeoDataFrame(vor_pts.append(local_gdf, ignore_index=True))
        elif(gen_type == 0):
            gdf_out = vor_pts
        else:
            gdf_out = local_gdf

        gdf_out = gdf_out.set_crs(epsg=3857)
        gdf_out = gdf_out.to_crs(epsg=4326)

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


    # Plotting of generated points and Voronoi regions
    if(plot or to_png):
        # create fig and axes to plot and compare points and Voronoi buffer regions

        # Setting plot title
        title = "Random Spatial Data - Radial Voronoi Generation\n" \
                "{} total points at ratio {} : Primary Points = {}, Secondary Points = {}\n".format(total_pts, ratio, bulk_points, local_points)
        if (rand_centroid):
            title += "Using moving centroid\n"
        else:
            title += "Using original centroid\n"

        # Local Generation type
        if (gen_type == 0):
            title += "Local Generation: None\n"
        elif (gen_type == 1):
            title += "Local Generation: Equal-area Voronoi, {} local regions\n".format(vor_num)
        elif (gen_type == 2):
            title += "Local Generation: Variable-area Voronoi - Points by Area, {} local regions\n".format(
                vor_num)
        elif (gen_type == 3):
            title += "Local Generation: Variable-area Voronoi - Equal Points, {} local regions\n".format(vor_num)
        else:
            title += "Local Generation: Error"
        if(breakdown):
            fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(13, 5))

            source.plot(ax=ax1, color='gray')
            source.plot(ax=ax2, color='gray')
            source.plot(ax=ax3, color='gray')

            vor_union.plot(ax=ax1, cmap='Blues', edgecolor='white', alpha=0.6)
            if gen_type != 0:
                local_vor_polygons.plot(ax=ax2, cmap='Blues', edgecolor='white', alpha=0.4)

            # vor_polygons.plot(ax=ax1, column='class', cmap='Blues', edgecolor='white', alpha=0.8)
            # centroid_point.plot(ax=ax1, color='Red')

            # plot the Bulk points
            if(bulk_points > 0):
                vor_pts.plot(ax=ax1, markersize=.5, color='black')
                vor_pts.plot(ax=ax3, markersize=.5, color='black')

            if gen_type != 0:
                local_gdf.plot(ax=ax2, markersize=0.5, color='white')
                local_gdf.plot(ax=ax3, markersize=0.5, color='black')

            fig.suptitle(title)  # Plot title text
            ax1.set_title("Primary Generation",y=0, pad=-14)
            ax1.axis("off")
            ax2.set_title("Secondary Generation",y=0, pad=-14)
            ax2.axis("off")
            ax3.set_title("Final Points generation", y=0.18, pad=-14)
            ax3.axis("off")
            #ax4.axis("off")
            plt.axis('equal')

        else:
            fig, ax = plt.subplots(figsize=(8,6))
            source = source.to_crs(epsg=3857)
            gdf_plot = gdf_out.to_crs(epsg=3857)

            if basemap:
                source.plot(ax=ax, facecolor="none", edgecolor='red')
                gdf_plot.plot(ax=ax, markersize=1, color='red', edgecolor='black')
                cx.add_basemap(ax)
            else:
                source.plot(ax=ax, color='gray')
                gdf_plot.plot(ax=ax, markersize=1, color='red')

            fig.suptitle(title)  # Plot title text
            ax.axis("off")
            plt.axis('equal')

        if(to_png):
            plt.savefig(f"{filename.split('.')[0]}_points{png_filename}.png")
        if(plot):
            plt.show()


    if(extra_var):
        if(isinstance(extra_var_file, str)):
            gdf_out[f'{extra_var_name}'] = csv_att(extra_var_file, total_pts)
        elif(isinstance(extra_var_file, list)):
            for i in range(len(extra_var_name)):
                gdf_out[f'{extra_var_name[i]}'] = csv_att(extra_var_file[i], total_pts)
        else:
            print("Extra var input invalid")

    # Exporting the generated points to an SQL file
    if(to_sql):
        gdf_to_sql(f"{filename.split('.')[0]}", gdf_out, total_pts, extra_var, extra_var_name)
    # Exporting the generated points to a GeoJSON file
    if(to_geojson):
        gdf_out.insert(0, 'PKID', range(0, len(gdf_out)))
        gdf_out.to_file(f"{filename.split('.')[0]}_points_4326.geojson", driver='GeoJSON')
        print("Successfully created GeoJSON file {}_points_4326.geojson with {} points".format(filename.split('.')[0], total_pts))

radial_spatial_points()

# radial_spatial_points uses the JSON parameter file
# radial_points_gen is the same function but with function parameters