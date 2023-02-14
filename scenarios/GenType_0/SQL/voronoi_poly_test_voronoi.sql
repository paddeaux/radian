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
INSERT into voronoi_poly_test (thegeom, dist_to_centre, poly_class) VALUES (ST_SetSRID(ST_PolygonFromText('POLYGON ((-203.340614 6715672.402842, -3504.789695 6705442.317886, -3854.650906 6705320.855966, -11566.767435 6709147.158657, -14167.799913 6721696.216514, -3066.563964 6720036.562284, -203.340614 6715672.402842))'),3857), 9, 6405.064523412688); 