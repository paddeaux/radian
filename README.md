# rand-geo-data
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
