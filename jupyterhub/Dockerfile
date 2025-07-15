FROM jupyterhub/jupyterhub:1.3

ARG ML_PROJECT_ID
ENV ML_PROJECT_ID=$ML_PROJECT_ID

ENV JPYTR_LISTEN_PORT='8000'
ENV JPYTR_USER='ssmuser'
ENV JPYTR_PASS='passw0rd'

RUN set -x \
  && pip install matplotlib notebook==6.5.7 numpy scikit-learn  markupsafe==2.0.1 \
  && mkdir -p /etc/jupyterhub \
  && cd       /etc/jupyterhub \
  && jupyterhub --generate-config -f ./jupyterhub_config.py \
  && cd -

COPY ./jupyterhub/ssmdemo.ipynb         /srv/jupyterhub/notebooks/
COPY ./jupyterhub/jupyterhub_config.py  /etc/jupyterhub/
COPY ./jupyterhub/docker-entrypoint.sh  /

RUN apt-get update 
RUN apt-get install -y build-essential libpq-dev python3-dev 
RUN pip install psycopg2
RUN pip install traitlets==5.1.1

COPY . /code
RUN cd /code \
    && pip install .

WORKDIR /srv/jupyterhub
ENTRYPOINT ["/docker-entrypoint.sh"]



