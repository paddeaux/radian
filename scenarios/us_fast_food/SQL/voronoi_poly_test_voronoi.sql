-- Voronoi polygon regions exported to SQL from the RADIAN Spatal Data Generator

DROP TABLE IF EXISTS voronoi_poly_test; 

CREATE TABLE voronoi_poly_test ( 
	pkid SERIAL PRIMARY KEY NOT NULL, 
	thegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326), 
	dist_to_centre NUMERIC,
	poly_class INTEGER

); 

-- Spatial index is now created

CREATE INDEX voronoi_poly_test_spatial_index ON voronoi_poly_test USING gist (thegeom); 
INSERT into voronoi_poly_test (thegeom, dist_to_centre, poly_class) VALUES (ST_SetSRID(ST_PolygonFromText('POLYGON ((-95.07480603636185 33.78440673693101, -100.38079068875706 35.99478784232729, -99.60097237539287 41.78489844820798, -96.22782289323298 42.28159740675499, -90.57219997297024 38.93516048609164, -95.07480603636185 33.78440673693101))'),3857), 425962.07152410626, 1); 