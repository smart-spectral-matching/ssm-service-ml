
FROM jupyterhub/jupyterhub

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

WORKDIR /srv/jupyterhub
ENTRYPOINT ["/docker-entrypoint.sh"]


