# MSc Data Science & Analytics Project - Random Spatial Data Generation with Python
This repo holds everything for my MSc in Data Science project. The project involves the creation of a Python tool to generate realistic random spatial data for use in assessment

## Running the Software
Running the tool will require installation of the necessary third-party packages including geopandas. The `msc_pg.py` file must be run in the same directory as the `parameters.json` file. The description and expected values for the parameters are as follows:
* `filename` : This will be the name/directory of the .geojson polygon boundary within which we wish to generate points
* `total_pts` : The total number of points to be generated within the polygon, approx. 1000-3000 recommended
* `gen_type` : Controls the type of local-level generation to be used. Expecting a number between 0 and 3:
    * 0 for just primary generation
    * 1 for local generation within voronoi regions of approximately equal area, each with an equal proportion of points
    * 2 for local generation within voronoi regions of variable area, each allocated a proporation relative to their size, larger polygons with more points than smaller ones
    * 3 for local generation within voronoi regions of variable area, each with an equal proportion of points
* `ratio` : Value between 0 and 1 that indicates the ratio of macro to micro points. 1 would result in 100% macro points, 0.5 would result in 50% macro and 50% micro etc.
* `vor_num` : Number of local regions to be generated for local point generation
* `rand_centroid` : True or false, indicates whether a moving random centroid is used in the primary generation or if the original polygon centroid is used.
* `int_range` : A list with length two, indicating the range of the random integers to be generated and assigned to each point, e.g. [1,100]
* `string_len` : Integer value that specifies the length of the random string to be generated and assigned to each point.
* `timestamp_range` : A list with length two, indicating the range within the random timestamp for each point is to be generated, e.g. ["2022-01-01 00:00:00", "2022-12-31 23:59:59"]
* `to_sql` : True/False. Indicates if final dataset will be outputted to an SQL file
* `to_geojson` : True/False. Indicates if final dataset will be outputted to a GeoJSON file
* `to_png` : True/False. Indicates if plots of generated points will be exported in PNG format
* `png_filename` : Specifies filename of above PNG file
* `plot` : True/False, Indicates if the final dataset and boundary will be plotted after generation using matplotlib
* `breakdown` : True/False. Indicates if the above plot will include a breakdown including 4 sepearate plots showing the macro, micro and final generation seperately.
* `extra_var` : True/False. Indicates if extra point columns from inputted CSV files will be included in dataset.
* `extra_var_name` : List of strings indicating the variable name(s) for the extra CSV columns to be added.
* `extra_var_file` : List of strings indicating the file names of the CSVs used for the extra variables.
* `set_seed` : True/False, indicates a seed will be used for the generation, allows for reproducibility. 
* `seed` : Integer seed value for random generation.

### Note on the CSV files
The CSV files for adding additional variables to the data should be simply formatted into two columns, one with a list of potential values, and a second column with weights indicating the frequency at which each value should occur in the final dataset.

## Section 1: Point Generation

Point generation occurs available at two scales **primary**, i.e. at an inputted main polygon level, and **secondary**, at a more localized level.

### Primary Generation

All gen types by default will utilize primary generation, with the `gen_type` parameter controlling the addition of secondary generation. 
Primary generation is carried out at the scale of the original source polygon, utilizing five voronoi-based buffers around the centroid in 
order to control generation. Each of these five regions are successively larger in size while also each being assigned an equal number of points. 
Thus resulting in the effect that points will concentrate towards the centre of the polygon with points density reducing further out towards the 
polygon boundaries. The primary method of generation utilizes Voronoi-based buffers to produce 5 concentric regions centred around the polygon 
centroid, with each of these regions being assigned an equal number of points to be generated. The result of this is points generation which 
is concentrated towards the given centroid, with the density of points decreasing the further away from the centroid, **producing an effect similar 
to that of a real life set of points in a metropolitan area**.

#### Primary Generation Algorithm

* The parameter file is read by the tool, and the source polygon is loaded
* Using the specified centroid type (true or random) a set of 256 voronoi regions is generated within the bounds of the source polygon
  * The distance from each of the voronoi region centroids to the centroid of the source polygon is calculated.
  * Each voronoi is then assigned to one of five classes to represent the region around the centroid that the voronoi polygon belongs to
  * The voronoi polygons are unioned by their class in order to produce five regions, centred around the centroid, of increasing size.
  * These are the Voronoi-based buffers
* The `ratio` parameter will determine the amount of points to be assigned to primary and secondary generation
* The tool will first perform the secondary generation (if `gen_type` is > 0)
  * A new set of Voronoi polygons (of an amount specified by the `vor_num` parameter) is created within the source polygon
  * The specified secondary generation method is then applied iteratively to each of the generated secondary voronoi regions
* Next the primary generation will take place:
  * Stat at the first Voronoi-based buffer polygon
  * Use the random point generation function, with the specified `gen_type`:
    * Generate a circular buffer around the centroid point, with a radius 1/5th the size of the max bounds of the source polygon
    * A fifth of the total primary points assinged will be generated iteratively
    * If the point lies within both the current buffer and the source polygon, it is appended to a list
    * Repeat the above step until all points for the current region have been generated
    * Move to the next circular buffer and repeat the above
  * Repeat the above step for each successive Voronoi-based buffer.
* Merge the primary and secondary (if applicable) datasets into one single GeoDataFrame
* Perform generation of random interger, string, and timestamp values and append values to each point in the gdf.
* Generate addition metadata values from CSV files and append to each point in the gdf.
* Output the file in whatever format specified, i.e. SQL, GeoJSON, Matplotlib Plot, PNG

#### Moving Centroid
The above generation can be performed using the original true polygon centroid, or through the generation of a "moving centroid" 
which is a randomly generated centroid in an eliptical area around the true centroid. This will in turn cause the resulting Voronoi-buffer 
regions to be shifted in the x/y axis according to the position of the generated moving centroid in relation to the original centroid. 
**This reflects the real world fact of the administrative or metropolitan centre of an area not necessarily being located at the exact 
geographic centre of the region.**

### Secondary Generation:
The function allows for more granular points generation through the addition of local-level generation in multiple ways.

#### 0) No Secondary Generation:
The function allows for just Macro-level generation, by setting the corresponding local parameters to zero, thus removing the local generation entirely

#### 1) Equal Area Secondary Generation:
The equal area local generation generates approximately equal-area Voronoi regions inside the source Polygon and then generates an equal proportion of points in each of these local polygons.

#### 2) Variable Area Secondary Generation: Equal Points
This generation produces Voronoi regions of varying size, with smaller regions concentrated towards the centroid of the polygon. These regions are assigned an equal number of points in each.

#### 3) Variable Area Local Secondary: Points by Area
This generation produces Voronoi regions of varying size, with smaller regions concentrated towards the centroid of the polygon. These regions are assigned points based on the area of each region, i.e. regions with larger area are allocated more points than smaller regions.

## Section 2: Output

The output feature of the tool is incredibly important as it allows students to be able to freely make use of the generated dataset. As a pre-processing step before any file export, the final points dataset is stored as a GeoPandas GeoDataframe.

### 1) GeoJSON
Exporting of the data to GeoJSON format is very simple thanks to the extensive use of the GeoPandas package, which includes simplified exporting of GeoDataFrames to GeoJSON with a built in function.

### 2) SQL
The exporting to SQL requires a great deal more thought than the exporting to geojson. GeoPandas has native compatibility for GeoJSON, allowing data to be exported in this way at the call of a function. The same can not be said for SQL. Through some simple manipulation of the columns and some checks for the data types, an SQL file can be written which will contain the table creation and insert statements for a PostgreSQL database of the generated spatial points.

### 3) Plotting
The tool offers the ability to display the outputted points on a Matplotlib map to demonstrate both the final set of points as well as the intermediate steps and buffers/polygons generated to reach that last dataframe.


## Planned Functionality:

* Generation of realistic random spatial data:
  * Data points to be distributed in a random fashion that behaves like real-world spatial data points (patterns, clustering, distance to roads/borders/etc.)
  * Points will have randomly generated realistic meta-data, such as names and various attributes, depending on the type of points being generated
* Exporting of data to a GeoJSON format
* Exporting of data to a PostgreSQL format


