DROP TABLE IF EXISTS argentina; 
CREATE TABLE argentina ( 
pkid SERIAL PRIMARY KEY NOT NULL, 
thegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326), 
rand_int INTEGER, 
rand_string VARCHAR, 
rand_ts TIMESTAMP,
var_string VARCHAR
); 
CREATE INDEX argentina_spatial_index ON argentina USING gist (thegeom); 
INSERT into argentina (thegeom, rand_int, rand_string, rand_ts, var_string ) VALUES (ST_SetSRID(ST_MakePoint(-66.69665223931464,-31.019890225169515),4326), 267, 'eLH2DW4jgSqYlYF', '2022-10-28 01:48:49', 'Nando''s'); 