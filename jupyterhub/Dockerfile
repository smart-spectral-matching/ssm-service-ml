
FROM jupyterhub/jupyterhub

ENV PORT_LISTEN='8914'

RUN set -x \
  && pip install matplotlib notebook numpy sklearn \
  \
  && mkdir -p /etc/jupyterhub && cd $_ \
  && openssl req \
    -x509 -nodes -days 365 -newkey rsa:2048 \
    -subj "/C=US/ST=Tennessee/L=Oak Ridge/O=Oak Ridge National Laboratory/OU=Research Software Engineering Group/CN=ssm-ml" \
    -keyout  server.key \
    -out     server.cert \
  && jupyterhub --generate-config -f ./jupyterhub_config.py \
  && mkdir -p /srv/jupyterhub/notebooks \
  && cd -

COPY ./ssmdemo.ipynb         /srv/jupyterhub/notebooks/
COPY ./jupyterhub_config.py  /etc/jupyterhub/

EXPOSE ${PORT_LISTEN}/tcp

CMD [ "jupyterhub", "--port", "${PORT_LISTEN}", "--ssl-key", "/etc/jupyterhub/server.key", "--ssl-cert", "/etc/jupyterhub/server.cert", "--debug" ]

