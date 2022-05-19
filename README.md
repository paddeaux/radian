# MSc Data Science & Analytics Project - Random Spatial Data Generation with Python
This repo holds everything for my MSc in Data Science project. The project involves the creation of a Python tool to generate realistic random spatial data for use in assessment

## Planned Functionality:

* Generation of realistic random spatial data:
  * Data points to be distributed in a random fashion that behaves like real-world spatial data points (patterns, clustering, distance to roads/borders/etc.)
  * Points will have randomly generated realistic meta-data, such as names and various attributes, depending on the type of points being generated
* Exporting of data to a GeoJSON format
* Exporting of data to a PostgreSQL format

## Current Goals & Progress

* Analyze Open Street Map attribute/location data
  * Using word frequency analysis in order to determine the most commonly used terms/words that appear in metadata
    * Bars/Restuarants:
      * Names
      * Types of food
      * Contact details
    * Shops:
      * Names
      * Types of goods sold
      * Contact details
    * Schools:
      * Names
      * Level of school (primary/secondary/tertiary)
      * Mixed or not
    * Transport Stops (Bus/Train)
      * Routes served
      * Stop code
  * Use these frequencies to produce dictionaries that will allow for generation of random realistic attribute data
  * Use distributions in order to allow metadata to be distributed inside the polygon in a realistic manner (e.g. restaurants having popular chain restaurants with multiple locations along with many one-off spots.
  * Analyse the distribution of the geographic locations of particular types of points (e.g. restaurants, shops, bus stops etc. etc.) in order to obtain a distribution with which to generate corresponding realistic random points which are located in a realistic manner
* Allow exporting of data points to a GeoJSON format
* Allow exporting of data for use in PostgreSQL

## Section 1: Point Generation

The core problem relating to this project is how to actually generate the points in a random, but realistic manner. The previous final year project which preceded my project used a simple random generator from a uniform distribution, meaning points are uniformly distributed throughout the given polygon. In the real world however points are seldom distributed this way, they cluster around certain areas be that as a result of admiistrative borders, roads, proximity to urban areas or proximity to other locations of the same type.

The following are methods or considerations that can be made for this generation:
* Using a Voronoi to split the polygon up into smaller polygons:
  * This could be used to generate points procedurally district by district
  * Or could also be used to develop a fake road network that could help with the weighting of points
  * Points could be generated at a full polygon level, then between Voronoi polygons where n = 2, then where n =4 and so on....
* The centroid of the polygon could be used to influence the distribution of points being generated, i.e. more points located towards the centroid than towards the outter limits of the polygon
* 

### Thus Far the Following Progress has been made on points generation:

#### Generating points around the polygon centroid
A Python function, `points_from_centroid` takes in a given polygon along with the number of points to be generated. The function then creates circular districts around the centroid of the polygon, with each being larger than the previous. Each circle is assigned the same proportion of points to be generated, resulting in the larger circles having lower density compared to the smaller inner circles. The net effect of this generation is that points appear to concentrate towards the centre of the polygon, with the density reducing the further from the polygon centroid.
   * This is good progress - however tests with large values of *N*, where *N* is the number of points to be generated, reveals the circular boundaries of the buffers created to generate points.
   * Additional steps need to be taken to remove this uniformity from the generation - perhaps adding in an extra variable at generation that would randomly assign an additional value to the generated x or y coordinates would serve to lessen the forming of this pattern.

#### Using Kmeans Clustering to generate
A function which uses uniform points generation to then implement Kmeans clustering to produce clusters and cluster centroids. This is made with the purposes of creating centroids for use in Voronoi generation. Using this method to generate these Voronoi centroids results in Voronoi polygons that are roughly equal in area.

#### Voronoi Generation
As stated above, the voronoi package allows for the generation/plotting of voronoi regions in a polygon. The method uses the Kmeans generated centroids as the basis for the generation of corresponding Voronoi polygons.

#### All of the above
Currently the program is able to take in a set number of points, *N*, and split that into *x* and *y*, where *x* is the number of points generated around the centroid of the full polygon itself as outlined above. *y* is the number of points that will then be distributed amongst the Voronoi regions generated above, with each Voronoi having $y/i$ points generated around their centroids, where *i* is the number of Voronoi regions to be generated

## Section 2: Restaurant Name Distribution

Restaurants of all shapes, types, and sizes appear all over the world. Cities will have collections of one-off, "Mom & Pop" spots, smaller chains of 2 or more restaurants as well as the franchises like McDonald's or Subway. In generating realistic random data, it will be helpful to observe the distribution of restaurants in order to mirror the distribution in the generated data.

The `franchises.py` file does exactly this. It has functionality to query the OSM API (though this is a WIP) as well as .csv files containing data exported from OSM (using Overpass Turbo). The file has functions to analyze the names of restaurants and output the proportions of locations that are one off, chains (2 or more), and franchises (5 or more). 
* Interestingly, checking across 5 sepearate cities, Dublin, LA, Chicago, London, and New York, the proportion of once-off restaraunts and chains is about 9:1. More cities will be tested to further validate this proportion, it could be useful to also perform this analysis on smaller types of city, allowing perhaps for different "archetypes" of location to be generated.
* This file will include a function that will generate a words dictionary with which to generate fake location names, and provide weights to these names later so that they appear on a correct proportion of locations, eg. 10% result in chains/franchises of fake restaurant names.
