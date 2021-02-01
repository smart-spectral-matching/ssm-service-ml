
FROM jupyterhub/jupyterhub

USER ssmuser
WORKDIR /home/ssmuser

ENV USERNAME=ssmuser
ENV PASSWORD=ssm2020!

ADD ./jupyterhub_config.py /home/jovyan/.jupyter/jupyterhub_config.py

RUN set -x \
  && openssl req \
    -x509 -nodes -days 365 -newkey rsa:2048 \
    -subj "/C=US/ST=Tennessee/L=Oak Ridge/O=Oak Ridge National Laboratory/OU=Research Software Engineering Group/CN=ssm-ml" \
    -keyout  jupyter.key \
    -out     jupyter.cert \
  \
  && pip install \
    matplotlib notebook numpy sklearn \
  \
  && useradd -m -p $(openssl passwd -1 ${PASSWORD}) -s /bin/bash -G sudo ${USERNAME} \
  && mkdir -p /home/ssmuser/notebooks

ADD ./ssmdemo.ipynb /home/ssmuser/notebooks/ssmdemo.ipynb

CMD [ "jupyterhub", "-f", "/home/jovyan/.jupyter/jupyterhub_config.py", "--debug" ]

