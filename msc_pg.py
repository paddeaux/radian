# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:54:29 2022

@author: paddy
"""
import random
from random import randint
from random import randrange
import datetime
import string
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    gdf_proj = vor_centroids.to_crs(poly.crs)

    # Convert the boundary geometry into a union of the polygon
    boundary_shape = cascaded_union(poly.geometry)
    coords = points_to_coords(gdf_proj.geometry)

    # Calculating the voronoi regions
    region_polys, region_pts = voronoi_regions_from_coords(coords, boundary_shape)

    # Create GeoDataFrame of the Voronoi Polygons
    df = pd.DataFrame.from_dict(region_polys, orient='index', columns=['geometry'])
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

def gdf_to_sql(table_name, gdf, num_rows):
    # Opens up an SQL file based on the table name, writes to the file and closes it
    sqlFile = open(f'{table_name}_SQL.sql', "w")
    sqlFile.write("")
    sqlFile.close()

    # Opens up the SQL file to append lines to it
    sqlFile = open(f'{table_name}_SQL.sql', "a")

    # SQL statments to create the table as well as drop if exists the table are appended
    sqlFile.write("DROP TABLE IF EXISTS {}; \n".format(table_name))
    sqlFile.write("CREATE TABLE {} ( \n".format(table_name))
    sqlFile.write("pkid SERIAL PRIMARY KEY NOT NULL, \n")
    sqlFile.write("thegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326), \n")
    sqlFile.write("rand_int INTEGER, \n")
    sqlFile.write("rand_string VARCHAR\n")
    sqlFile.write("); \n")

    # Creation of Spatial Index for the SQL file
    sqlFile.write("CREATE INDEX {}_spatial_index ON {} USING gist (thegeom); \n".format(table_name, table_name))

    # Loop through every point in the GeoDataFrame to write an instert statement to append to the SQL file
    for row in gdf.iterrows():
        # Set 'y' value to Latitude and 'x' value to Longitude.
        lat = row[1][0].y
        lon = row[1][0].x
        # Pull the randomly generated strings and ints from the dataframe
        rand_str = row[1][1]
        rand_int = row[1][2]

        # Insert statement for each point along with the included variables
        query = "INSERT into {} (thegeom, rand_int, rand_string) VALUES (ST_SetSRID(ST_MakePoint({},{}),4326), {}, '{}'); \r".format(
            table_name, lon, lat, rand_int, rand_str)
        # Write query string to SQL file
        sqlFile.write(query)

    print("Successfully printed {} rows to {} with table name: {}.".format(num_rows, f'{table_name}_SQL.sql', table_name))

# This function takes in a Shapely Polygon, number of points to be generated, and generation type, and produces
# a set of random points within that polygon, generated in a pattern set by the generation type:
# 'uni' = Points are uniformly distributed within the polygon, meaning there is no clustering or other patterns
# 'rand' = Points are generated around a randomly generated "moving centroid", with points concentrated around the centroid
# 'cent' = Points are generated around the polygon centroid, with points centred around it

def random_point_gen(poly, num_points, gen_type):
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
        section_pts = num_points / section_num

        # Values used to shift point locations to account for the moving centroid generation
        cent_diff_x = (cx - centroid_point.x) / section_num
        cent_diff_y = (cy - centroid_point.y) / section_num

        # Points are generated section by section, along with randomly generated int and string variables
        for i in range(0, section_num):

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
                        random_string = ''.join(
                            random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
                        # Create random int with bounds inputted from .ini file.
                        random_int = randint(0, 1000)
                        # Take a random number, translate that to seconds and develops that into HH:MM:SS format.
                        random_timestamp = datetime.timedelta(seconds=randrange(86400))
                        points.append(random_point)
                        rand_strings.append(random_string)
                        rand_ints.append(random_int)
                        # rand_ts.append(random_timestamp)
                        current_pts += 1
                if (current_pts >= section_pts):
                    section_num += 1

            buffers.append(c_current)

        # GeoDataFrame of the circular buffers is created (optional use for visualization)
        df_buff = pd.DataFrame(buffers, columns=['geometry'])
        gdf_buff = gpd.GeoDataFrame(df_buff, geometry='geometry')

        # Points list is converted to a GeoDataFrame and outputted
        df = pd.DataFrame(list(zip(points, rand_strings, rand_ints)),
                          columns=['geometry', 'random_string', 'random_int'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf.crs = poly._crs
        return gdf

    # Original centroid generation
    elif (gen_type == 'cent'):

        section_num = 5
        section_size = (radius * 0.8) / section_num
        section_pts = num_points / section_num
        for i in range(0, section_num):
            c_current = poly.centroid.buffer(section_size * (i+1))
            current_pts = 0
            while current_pts < section_pts:
                random_point = Point([random.uniform(min_x + random.uniform(0,100), max_x - random.uniform(0,100)), random.uniform(min_y + random.uniform(0,100), max_y - random.uniform(0,100))])
                if (random_point.within(poly)):
                    if (random_point.within(c_current)):
                        random_string = ''.join(
                            random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
                        random_int = randint(0, 1000)
                        random_timestamp = datetime.timedelta(seconds=randrange(86400))
                        points.append(random_point)
                        rand_strings.append(random_string)
                        rand_ints.append(random_int)
                        current_pts += 1
                if (current_pts >= section_pts):
                    section_num += 1
            buffers.append(c_current)
        df_buff = pd.DataFrame(buffers, columns=['geometry'])
        gdf_buff = gpd.GeoDataFrame(df_buff, geometry='geometry')
        df = pd.DataFrame(list(zip(points, rand_strings, rand_ints)), columns=['geometry', 'random_string', 'random_int'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        return gdf

def radial_rpg(filename, total_pts, local_gen_type, local_ratio, local_vor_num, rand_centroid, to_sql, to_geojson, to_png, plot, animate):
    # Reading in the GeoJSON file and setting the CRS to a meter-based projection
    source = gpd.read_file(filename)
    source = source.to_crs(epsg=3857)
    min_x, min_y, max_x, max_y = source.total_bounds

    # Set the number used for Voronoi-based buffer generation to 256
    vor_num = 256

    # This set of if-else statements determines how the Voronoi polygons are generated
    if(rand_centroid):
        print("Generating Voronoi with Moving Centroid...")
        vor_polygons = voronoi_gen(source, vor_num, 'rand')
    else:
        print("Generating Voronoi with Original...")
        vor_polygons = voronoi_gen(source, vor_num, 'eq')

    vor_polygons.crs = source.crs

    # Source-boundary Generation
    bulk_points = total_pts * local_ratio

    #### Local level generation ###

    # local_gen_type:
        # 0 for no local-level generation
        # 1 for Equal-area Voronoi local generation
        # 2 for Variable-area Voronoi local generation with points determined by area
        # 3 for Variable-area Voronoi local generation with equal points in each Voronoi

    # Set no local generation as the default
    if local_gen_type > 3:
        local_gen_type = 0

    # Local generation with approximately equal-area Voronoi polygons
    if local_gen_type == 1:
        if local_vor_num > 256:
            local_vor_num = 256
            print("Max local_vor_num is 256!")
        #bulk_points = total_pts * local_ratio
        local_points = total_pts * (1 - local_ratio)
        local_vor_points = local_points / local_vor_num

        local_vor_polygons = voronoi_gen(source, local_vor_num, 'eq')
        print("Voronoi complete.")
        # the polyogn crs is set as the main polygon crs
        local_vor_polygons.crs = source.crs

        local_gdf = gpd.GeoDataFrame()
        for i in range(0, local_vor_num):
            current = random_point_gen(local_vor_polygons['geometry'][i], local_vor_points, 'cent')
            local_gdf = gpd.GeoDataFrame(local_gdf.append(current, ignore_index=True ))

    # Local generation with variable area Voronoi polygons with number of points based on area
    elif local_gen_type == 2:
        if local_vor_num > 32:
            local_vor_num = 32
            print("Max poly_area value is 32!")
        #bulk_points = total_pts * local_ratio
        local_points = total_pts * (1-local_ratio)

        print("Generating Voronoi with Variable Area...")
        local_vor_polygons = voronoi_gen(source, local_vor_num, 'area')

        # calculating the area of each polygon to determine the proportion of points in each
        local_area_union = local_vor_polygons.dissolve()
        local_area = local_area_union.area

        local_gdf = gpd.GeoDataFrame()
        for i in range(0, local_vor_num):
            area_prop = local_vor_polygons['geometry'][i].area / local_area
            current_local_points = int(round(local_points * area_prop))
            current = random_point_gen(local_vor_polygons['geometry'][i], current_local_points, 'cent')
            local_gdf = gpd.GeoDataFrame(local_gdf.append(current, ignore_index=True))

    # Local generation with variable area Voronoi polygons with equal number of points in each
    elif local_gen_type == 3:
        if local_vor_num > 32:
            local_vor_num = 32
            print("Max poly_area value is 32!")
        local_points = total_pts * (1 - local_ratio)
        local_vor_points = local_points / local_vor_num

        print("Generating Voronoi with Variable Area...")
        local_vor_polygons = voronoi_gen(source, local_vor_num, 'area')

        print("Voronoi complete.")
        # the polyogn crs is set as the main polygon crs
        local_vor_polygons.crs = source.crs

        local_gdf = gpd.GeoDataFrame()
        for i in range(0, local_vor_num):
            current = random_point_gen(local_vor_polygons['geometry'][i], local_vor_points, 'cent')
            local_gdf = gpd.GeoDataFrame(local_gdf.append(current, ignore_index=True))

    # Bulk generation in the source polygon
    if(bulk_points > 0):
        # dissolve the vor_polygons by the class, determined by their distance to the centre of the polygon
        vor_union = vor_polygons.dissolve(by='class', as_index=False)
        # combining successive Voronoi regions to allow points generating inside them
        vor_all = gpd.GeoDataFrame()
        for i in range(len(vor_union['geometry'])):
            current = gpd.GeoSeries(cascaded_union(list(vor_union['geometry'][0:i + 1])))
            vor_all = gpd.GeoDataFrame(vor_all.append(current, ignore_index=True))

        # Generating points inside the merged Voronoi regions
        vor_points = total_pts / 5
        vor_pts = gpd.GeoDataFrame()
        for i in range(len(vor_all[0])):
            ### should this be the rand???
            if(rand_centroid):
                gdf = random_point_gen(vor_all[0][i], vor_points, 'rand')
            else:
                gdf = random_point_gen(vor_all[0][i], vor_points, 'cent')

            vor_pts = gpd.GeoDataFrame(vor_pts.append(gdf, ignore_index=True))

    # Merging the bulk and local point dataframes for output to SQL or GeoJSON
    if(to_sql or to_geojson):
        if(local_gen_type != 0 and bulk_points > 0):
            gdf_out = gpd.GeoDataFrame(vor_pts.append(local_gdf, ignore_index=True))
        elif(local_gen_type == 0):
            gdf_out = vor_pts
        else:
            gdf_out = local_gdf

        gdf_out = gdf_out.set_crs(epsg=3857)
        gdf_out = gdf_out.to_crs(epsg=4326)

    # Plotting of generated points and Voronoi regions
    if(plot or to_png):
        # create fig and axes to plot and compare points and Voronoi buffer regions
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

        source.plot(ax=ax1, color='gray')
        source.plot(ax=ax2, color='gray')
        source.plot(ax=ax3, color='gray')
        source.plot(ax=ax4, color='gray')

        vor_union.plot(ax=ax1, cmap='Blues', edgecolor='white', alpha=0.8)
        if local_gen_type != 0:
            local_vor_polygons.plot(ax=ax3, cmap='Blues', edgecolor='white', alpha=0.4)

        # vor_polygons.plot(ax=ax1, column='class', cmap='Blues', edgecolor='white', alpha=0.8)
        # centroid_point.plot(ax=ax1, color='Red')

        # plot the Bulk points
        if(bulk_points > 0):
            #vor_pts.plot(ax=ax1, markersize=.5, color='black')
            vor_pts.plot(ax=ax2, markersize=.5, color='black')

        if local_gen_type != 0:
            local_gdf.plot(ax=ax3, markersize=0.5, color='white')
            local_gdf.plot(ax=ax2, markersize=0.5, color='black')

        # set figure title
        fig_title = "Points Generation: " + str(total_pts) + " total points, Local Ratio of " + str(local_ratio) + "\n"
        if(rand_centroid):
            fig_title += "Moving Centroid "
        else:
            fig_title += "Fixed Centroid "


        fig.suptitle(fig_title)  # Plot title text
        ax1.set_title("Bulk points generation with Voronoi-based buffers")
        ax1.axis("off")
        ax2.set_title("Final Points generation")
        ax2.axis("off")
        ax3.set_title("Local Generation")
        ax3.axis("off")
        ax4.axis("off")

        plt.axis('equal')

        if(to_png):
            plt.savefig(f"{filename.split('.')[0]}_points.png")
        if(plot):
            plt.show()

    print(vor_pts['geometry'][0])

    if(animate):
        ani_points = vor_pts['geometry']
        fig, ax = plt.subplots(figsize=(8, 6))
        source.plot(ax=ax)

        def animate(i):
            current_pt = ani_points[i]
            #ani_points.drop(index=[0])
            print("Printing point...")
            print(current_pt)
            gpd.GeoSeries(current_pt).plot(ax=ax, markersize=1, color='black')

        ani = FuncAnimation(fig, animate, total_pts, interval=5, repeat=False)
        plt.show()

    # Exporting the generated points to an SQL file
    if(to_sql):
        gdf_to_sql(f"{filename.split('.')[0]}", gdf_out, total_pts)

    # Exporting the generated points to a GeoJSON file
    if(to_geojson):
        gdf_out.to_file(f"{filename.split('.')[0]}_points_4326.geojson", driver='GeoJSON')
        print("Successfully created GeoJSON file {}_points_4326.geojson with {} points".format(filename.split('.')[0],total_pts))

    print(vor_all)

def geometry_testing_plot(vor_num, total_pts):
    file_dirs = ['square.geojson', 'triangle.geojson', 'star.geojson', 's.geojson']
    for file in file_dirs:
        radial_rpg(
            filename='TestingGeoJSONs/{}'.format(file),
            total_pts=total_pts,
            local_gen_type=1,
            local_ratio=0.5,
            local_vor_num=vor_num,
            rand_centroid=True,
            to_sql=False,
            to_geojson=False,
            to_png=False,
            plot=True,
            animate=False
        )


def main():
    radial_rpg(
        filename='london.geojson',
        total_pts=3000,
        local_gen_type=2,
        local_ratio=0.7,
        local_vor_num=16,
        rand_centroid=True,
        to_sql=False,
        to_geojson=False,
        to_png=False,
        plot=True,
        animate=False
    )

main()
#geometry_testing_plot(16, 300)

# To do:
# Polygon is not iterable error (works after re-running)

