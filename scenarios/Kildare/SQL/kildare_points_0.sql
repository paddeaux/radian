DROP TABLE IF EXISTS kildare; 
CREATE TABLE kildare ( 
pkid SERIAL PRIMARY KEY NOT NULL, 
thegeom GEOMETRY DEFAULT ST_GeomFromText('POINT(0,51)', 4326), 
rand_int INTEGER, 
rand_string VARCHAR, 
rand_ts TIMESTAMP,
var_string VARCHAR,
var_number INTEGER,
var_number2 INTEGER
); 
CREATE INDEX kildare_spatial_index ON kildare USING gist (thegeom); 
INSERT into kildare (thegeom, rand_int, rand_string, rand_ts, var_string, var_number, var_number2 ) VALUES (ST_SetSRID(ST_MakePoint(-6.817857478197046,53.19150456097833),4326), 225, 'uFZVepmmMUnvnDN', '2022-09-02 18:48:30', 'Nando''s', 2, 1); 