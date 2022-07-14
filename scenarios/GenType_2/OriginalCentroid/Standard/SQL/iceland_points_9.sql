DROP TABLE IF EXISTS iceland; 
CREATE TABLE iceland ( 
pkid SERIAL PRIMARY KEY NOT NULL, 
thegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326), 
rand_int INTEGER, 
rand_string VARCHAR, 
rand_ts TIMESTAMP,
var_string VARCHAR
); 
CREATE INDEX iceland_spatial_index ON iceland USING gist (thegeom); 
INSERT into iceland (thegeom, rand_int, rand_string, rand_ts, var_string ) VALUES (ST_SetSRID(ST_MakePoint(-18.882949866701047,65.08075059123988),4326), 669, 'GARZEwvVS2ScxGe', '2022-05-09 21:14:17', 'Pret A Manger'); 