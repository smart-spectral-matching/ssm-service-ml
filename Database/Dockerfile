FROM code.ornl.gov:4567/rse/datastreams/ssm/backend/ssm-ml/postgres:13.2-alpine
ARG POSTGRES_PASSWORD 
ENV POSTGRES_PASSWORD=$POSTGRES_PASSWORD
ENV POSTGRES_USER=ssm
ENV DATABASE=ssm

ADD ssm.sql /docker-entrypoint-initdb.d/init.sql

