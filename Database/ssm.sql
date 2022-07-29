CREATE USER postgres PASSWORD 'postgres';
CREATE TABLE models (name varchar(255) PRIMARY KEY, datasets text, filter text, extrema text, labels text, description text, model bytea, f1 float, recall float, selectivity float, false_discovery float,  false_omission float, true_positive int, true_negative int, false_positive int, false_negative int, created timestamp default current_timestamp);
CREATE TABLE graphs (image bytea, model varchar(255));
GRANT CONNECT ON DATABASE ssm TO postgres;
GRANT SELECT, INSERT, UPDATE, DELETE ON models TO postgres;
GRANT SELECT, INSERT, UPDATE, DELETE ON graphs TO postgres;

