# MSc Data Science & Analytics Project - Random Spatial Data Generation with Python
This repo holds everything for my MSc in Data Science project. The project involves the creation of a Python tool to generate realistic random spatial data for use in assessment

## Planned Functionality:

* Generation of realistic random spatial data:
  * Data points to be distributed in a random fashion that behaves like real-world spatial data points (patterns, clustering, distance to roads/borders/etc.)
  * Points will have randomly generated realistic meta-data, such as names and various attributes, depending on the type of points being generated
* Exporting of data to a GeoJSON format
* Exporting of data to a PostgreSQL format

## Current Goals & Progress

* Use circular buffering to allow generation of points centred around the centroid of a polygon
* Allow exporting of data points to a GeoJSON format
* Allow exporting of data for use in PostgreSQL

### Optional Goals
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


## Section 1: Point Generation

### Macro Generation: Moving and True Centroid
A Python function has been created to produce randomly generated spatial data points, located inside a given Polygon in GeoJSON format. The function is able to produce multiple forms of random generationg using a mix of local and source-level generation.

The primary method of generation utilizes Voronoi-based buffers to produce 5 concentric regions centred around the polygon centroid, with each of these regions being assigned an equal number of points to be generated. The result of this is points generationg which is concentrated towards the given centroid, with the density of points decreasing the furhter away from the centroid, **producing an effect similar to that of a real life set of points in a metropolitan area**.

The above generation can be performed using the original true polygon centroid, or through the generation of a "moving centroid" which is a randomly generated centroid in an eliptical area around the true centroid. This will in turn cause the resulting Voronoi-buffer regions to be shifted in the x/y axis according to the position of the generated moving centroid in relation to the original centroid. **This reflects the real world fact of the administrative or metropolitan centre of a an area not necessarily being located at the exact geographic centre of the region.**

### Micro Generation:
The function allows for more granular points generation through the addition of local-level generation in multiple ways.

#### Equal Area Local Generation:
The equal area local generation generates approximately equal-area Voronoi regions inside the source Polygon and then generates an equal proportion of points in each of these local polygons.

#### Variable Area Local Generation


## Section 2: Restaurant Name Distribution

Restaurants of all shapes, types, and sizes appear all over the world. Cities will have collections of one-off, "Mom & Pop" spots, smaller chains of 2 or more restaurants as well as the franchises like McDonald's or Subway. In generating realistic random data, it will be helpful to observe the distribution of restaurants in order to mirror the distribution in the generated data.

The `franchises.py` file does exactly this. It has functionality to query the OSM API (though this is a WIP) as well as .csv files containing data exported from OSM (using Overpass Turbo). The file has functions to analyze the names of restaurants and output the proportions of locations that are one off, chains (2 or more), and franchises (5 or more). 
* Interestingly, checking across 5 sepearate cities, Dublin, LA, Chicago, London, and New York, the proportion of once-off restaraunts and chains is about 9:1. More cities will be tested to further validate this proportion, it could be useful to also perform this analysis on smaller types of city, allowing perhaps for different "archetypes" of location to be generated.
* This file will include a function that will generate a words dictionary with which to generate fake location names, and provide weights to these names later so that they appear on a correct proportion of locations, eg. 10% result in chains/franchises of fake restaurant names.
