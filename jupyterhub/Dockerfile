FROM jupyterhub/jupyterhub

#RUN jupyterhub generate-config
ADD ./jupyterhub_config.py /home/jovyan/.jupyter/jupyterhub_config.py
ADD ./jupyter.cert /home/jovyan/.jupyter/jupyter.cert
ADD ./jupyter.key /home/jovyan/.jupyter/jupyter.key

RUN pip install matplotlib notebook numpy sklearn


ENV USERNAME=${JUPYTER_USERNAME}
ENV PASSWORD=${JUPYTER_PASSWORD}

RUN useradd -m -p $(openssl passwd -1 ${PASSWORD}) -s /bin/bash -G sudo ${USERNAME}
USER ssmuser
WORKDIR /home/ssmuser

RUN mkdir /home/ssmuser/notebooks
ADD ./ssmdemo.ipynb /home/ssmuser/notebooks/ssmdemo.ipynb

CMD [ "jupyterhub", "-f", "/home/jovyan/.jupyter/jupyterhub_config.py", "--debug" ]
