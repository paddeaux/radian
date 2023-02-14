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
INSERT into voronoi_poly_test (thegeom, dist_to_centre, poly_class) VALUES (ST_SetSRID(ST_PolygonFromText('POLYGON ((-733732.807556 7053249.812991, -733170.622256 7053556.053029, -733498.565631 7054295.102515, -733757.632024 7054280.707789, -734091.087457 7053761.184716, -733732.807556 7053249.812991))'),3857), 14, 377.6030335866513); 