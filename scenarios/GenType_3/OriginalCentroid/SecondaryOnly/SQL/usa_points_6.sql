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
INSERT into usa (thegeom, rand_int, rand_string, rand_ts, var_string ) VALUES (ST_SetSRID(ST_MakePoint(-80.97825479302426,33.31034176293649),4326), 285, '0aWQeSO9yAwcMMi', '2022-05-03 08:37:13', 'Prezzo'); 