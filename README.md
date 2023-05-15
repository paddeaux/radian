# **RADIAN**: Synthetic Spatial Data in Python

## **Overview**

**RADIAN** (**RA**n**D**om spat**I**al d**A**ta ge**N**erator) is a *Python-based* tool to generate synthetic geographic datasets for classroom and teaching environments. RADIAN utilizes a unique *voronoi-based* buffering system in order to replicate the *radial* nature of many real-world spatial datasets. Given a polygon in `GeoJSON` format and the relevent `JSON` parameter file, RADIAN can export synthetic datasets in `GeoJSON` and `postgreSQL` formats.

This readme will briefly outline the use of the tool and some examples of what it is capable of. For a more in-depth exploration of the underlying algorithms, please consult the [wiki](https://github.com/paddeaux/msc_rng/wiki/RADIAN---Wiki).

A short demonstration video is available [here](https://maynoothuniversity-my.sharepoint.com/:v:/g/personal/patrick_gorry_2015_mumail_ie/ETvmj7NewVpNqyeULOhxhP4BOkQLp1oirA-WBtysssEpCw?e=3du5cF).

## **Running the Software**
Running the tool will require installation of the necessary third-party packages including geopandas. The `radian.py` file must be run in the same directory as the `parameters.json` file. A detailed description of the running parameters and their expected values/limits is available [here](https://github.com/paddeaux/msc_rng/wiki/Parameters).

The once the `parameters.json` file is listed in the same directory as `radian.py`, the tool can be executed by simply runningthe `radian.py` file from the command line.

## **Sample Run**

Let's say we wish to generate a synthetic dataset representing *fast food restaurants* in **Glasgow**. We will generate 400 points, each with a phone number, opening & closing time, area code, and finally the name of the restaurant. The parameters used for this generation can be viewed [here](https://github.com/paddeaux/msc_rng/blob/main/glasgow_example.json).

![RADIAN example](https://github.com/paddeaux/msc_rng/blob/main/images/glasgow_example.png?raw=true)







