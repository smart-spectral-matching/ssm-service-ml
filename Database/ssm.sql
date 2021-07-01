CREATE USER postgres PASSWORD 'postgres';
CREATE TABLE models (name varchar(255) PRIMARY KEY, datasets text, model bytea, created timestamp default current_timestamp);
GRANT CONNECT ON DATABASE ssm TO postgres;
GRANT SELECT, INSERT, UPDATE, DELETE ON models TO postgres;

