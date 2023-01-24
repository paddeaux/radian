DROP TABLE IF EXISTS ireland; 
CREATE TABLE ireland ( 
pkid SERIAL PRIMARY KEY NOT NULL, 
thegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326), 
rand_int INTEGER, 
rand_string VARCHAR, 
rand_ts TIMESTAMP
); 
CREATE INDEX ireland_spatial_index ON ireland USING gist (thegeom); 
INSERT into ireland (thegeom, rand_int, rand_string, rand_ts) VALUES (ST_SetSRID(ST_MakePoint(-7.166234247819084,53.34395762037963),4326), 676, 'XgyamwbSNfnXmRR', '2022-09-30 14:54:44'); 