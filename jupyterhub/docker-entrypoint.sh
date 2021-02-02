#!/bin/sh

OS_USER="${JPYTR_USER:-ss7user}"
OS_PASS="${JPYTR_PASS:-Larm33pa7}"

useradd \
    --no-create-home \
    --groups sudo \
  ${OS_USER}

echo '${OS_USER}:${OS_PASS}' | chpasswd

chown -R ${OS_USER}:${OS_USER} /srv/jupyterhub

chmod +r /etc/shadow

su - ${OS_USER} <<_EOT

jupyterhub --debug \
  --ip 0.0.0.0 \
  --port ${JPYTR_LISTEN_PORT:-8080} \
  --ssl-key   /etc/jupyterhub/server.key \
  --ssl-cert  /etc/jupyterhub/server.cert

_EOT

