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
INSERT into iceland (thegeom, rand_int, rand_string, rand_ts, var_string ) VALUES (ST_SetSRID(ST_MakePoint(-18.635707926028605,65.10287487234514),4326), 902, 'tfKSl15btygqc3K', '2022-07-16 18:43:24', 'KFC'); 