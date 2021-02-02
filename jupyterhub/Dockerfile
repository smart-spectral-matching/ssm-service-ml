
FROM jupyterhub/jupyterhub

ENV JPYTR_LISTEN_PORT='8914'
ENV JPYTR_USER='ssmuser'
ENV JPYTR_PASS='Larm33pa8'

RUN set -x \
  && pip install matplotlib notebook numpy sklearn \
  && mkdir -p /etc/jupyterhub \
  && cd       /etc/jupyterhub \
  && openssl req \
    -x509 -nodes -days 365 -newkey rsa:2048 \
    -subj "/C=US/ST=Tennessee/L=Oak Ridge/O=Oak Ridge National Laboratory/OU=Research Software Engineering Group/CN=ssm-ml" \
    -keyout  ./server.key \
    -out     ./server.cert \
  && chmod 644 ./server.key \
  && jupyterhub --generate-config -f ./jupyterhub_config.py \
  && cd -

COPY ./ssmdemo.ipynb         /srv/jupyterhub/notebooks/
COPY ./jupyterhub_config.py  /etc/jupyterhub/
COPY ./docker-entrypoint.sh  /

#EXPOSE ${PORT_LISTEN}/tcp

WORKDIR /srv/jupyterhub
ENTRYPOINT ["/docker-entrypoint.sh"]

