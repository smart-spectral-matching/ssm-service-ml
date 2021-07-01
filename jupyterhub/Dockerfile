FROM jupyterhub/jupyterhub

ARG ML_PROJECT_ID
ENV ML_PROJECT_ID=$ML_PROJECT_ID
ARG GITLAB_PASSWORD
ENV GITLAB_PASSWORD=$GITLAB_PASSWORD
ARG GITLAB_USERNAME
ENV GITLAB_USERNAME=$GITLAB_USERNAME

#RUN jupyterhub generate-config
ADD ./jupyterhub_config.py /home/jovyan/.jupyter/jupyterhub_config.py
ADD ./jupyter.cert /home/jovyan/.jupyter/jupyter.cert
ADD ./jupyter.key /home/jovyan/.jupyter/jupyter.key

RUN apt-get update 
RUN apt-get install -y build-essential libpq-dev python3-dev 
RUN pip install matplotlib notebook numpy psycopg2 sklearn 
RUN pip install ssm-ml --extra-index-url https://__token__:$GITLAB_PASSWORD@code.ornl.gov/api/v4/projects/$ML_PROJECT_ID/packages/pypi/simple

ENV USERNAME=ssmuser
ENV PASSWORD=ssm2020!

RUN useradd -m -p $(openssl passwd -1 ${PASSWORD}) -s /bin/bash -G sudo ${USERNAME}
USER ssmuser
WORKDIR /home/ssmuser

RUN mkdir /home/ssmuser/notebooks
ADD ./ssmdemo.ipynb /home/ssmuser/notebooks/ssmdemo.ipynb

CMD [ "jupyterhub", "-f", "/home/jovyan/.jupyter/jupyterhub_config.py", "--debug" ]