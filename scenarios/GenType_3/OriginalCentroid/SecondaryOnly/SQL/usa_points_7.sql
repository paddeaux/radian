DROP TABLE IF EXISTS usa; 
CREATE TABLE usa ( 
pkid SERIAL PRIMARY KEY NOT NULL, 
thegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326), 
rand_int INTEGER, 
rand_string VARCHAR, 
rand_ts TIMESTAMP,
var_string VARCHAR
); 
CREATE INDEX usa_spatial_index ON usa USING gist (thegeom); 
INSERT into usa (thegeom, rand_int, rand_string, rand_ts, var_string ) VALUES (ST_SetSRID(ST_MakePoint(-80.23851115400413,35.50044015089714),4326), 535, 'BhTSroPpUVLAQcK', '2022-07-28 22:02:47', 'McDonald''s'); 