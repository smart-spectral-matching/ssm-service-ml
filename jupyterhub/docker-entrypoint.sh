#!/bin/sh

# NOTE: This must match jupyterhub_config.py under c.LocalAuthenticator.group_whitelist
JPYTR_GROUP="ssm"

which jupyterhub || exit 1

# Setup user and password
useradd -M ${JPYTR_USER}
echo "${JPYTR_USER}:${JPYTR_PASS}" | chpasswd
chown -R ${JPYTR_USER} /srv/jupyterhub

# Setup group
groupadd ${JPYTR_GROUP}
adduser ${JPYTR_USER} ${JPYTR_GROUP}

jupyterhub --debug \
  --config    /etc/jupyterhub/jupyterhub_config.py \
  --port ${JPYTR_LISTEN_PORT} \
  --ip 0.0.0.0

