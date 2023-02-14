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
INSERT into voronoi_poly_test (thegeom, dist_to_centre, poly_class) VALUES (ST_SetSRID(ST_PolygonFromText('POLYGON ((-770065.636976 7010068.464117, -760863.40408 7005867.207163, -755457.396182 7009134.353497, -757200.066746 7020187.770172, -770306.831907 7017609.499564, -770065.636976 7010068.464117))'),3857), 10, 6567.571560042697); 