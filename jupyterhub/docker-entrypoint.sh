#!/bin/sh

which jupyterhub || exit 1

useradd -M ${JPYTR_USER}
echo "${JPYTR_USER}:${JPYTR_PASS}" | chpasswd

chown -R ${JPYTR_USER}:${JPYTR_PASS} /srv/jupyterhub
jupyterhub --debug \
  --config    /etc/jupyterhub/jupyterhub_config.py \
  --ssl-cert  /etc/jupyterhub/server.cert \
  --ssl-key   /etc/jupyterhub/server.key \
  --port ${JPYTR_LISTEN_PORT} \
  --ip 0.0.0.0

