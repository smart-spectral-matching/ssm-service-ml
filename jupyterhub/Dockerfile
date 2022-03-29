FROM code.ornl.gov:4567/rse/datastreams/ssm/backend/ssm-ml/jupyterhub/jupyterhub:1.3

ARG ML_PROJECT_ID
ENV ML_PROJECT_ID=$ML_PROJECT_ID
ARG GITLAB_PASSWORD
ENV GITLAB_PASSWORD=$GITLAB_PASSWORD
ARG GITLAB_USERNAME
ENV GITLAB_USERNAME=$GITLAB_USERNAME

ENV JPYTR_LISTEN_PORT='8000'
ENV JPYTR_USER='ssmuser'
ENV JPYTR_PASS='passw0rd'

RUN set -x \
  && pip install matplotlib notebook numpy sklearn \
  && mkdir -p /etc/jupyterhub \
  && cd       /etc/jupyterhub \
  && jupyterhub --generate-config -f ./jupyterhub_config.py \
  && cd -

COPY ./ssmdemo.ipynb         /srv/jupyterhub/notebooks/
COPY ./jupyterhub_config.py  /etc/jupyterhub/
COPY ./docker-entrypoint.sh  /

RUN apt-get update 
RUN apt-get install -y build-essential libpq-dev python3-dev 
RUN pip install matplotlib notebook numpy psycopg2 sklearn markupsafe==2.0.1
RUN pip install ssm-ml --extra-index-url https://__token__:$GITLAB_PASSWORD@code.ornl.gov/api/v4/projects/$ML_PROJECT_ID/packages/pypi/simple

WORKDIR /srv/jupyterhub
ENTRYPOINT ["/docker-entrypoint.sh"]



