DROP TABLE IF EXISTS usa; 
CREATE TABLE usa ( 
pkid SERIAL PRIMARY KEY NOT NULL, 
thegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326), 
rand_int INTEGER, 
rand_string VARCHAR, 
rand_ts TIMESTAMP,
fast_food_name VARCHAR
); 
CREATE INDEX usa_spatial_index ON usa USING gist (thegeom); 
INSERT into usa (thegeom, rand_int, rand_string, rand_ts, fast_food_name ) VALUES (ST_SetSRID(ST_MakePoint(-100.01404899763588,40.41329879040976),4326), 47619, '12UjzAEmKmZD3zm', '2022-08-24 07:47:24', 'Wienerschnitzel'); 