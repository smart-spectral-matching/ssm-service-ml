# SSM Machine Learning

This is the machine learning component for the Smart Spectral Matching (SSM) project.

## Getting Started

To launch the JupyterHub service

```
cd jupyterhub
docker build -t ssm-ml .
docker run ssm-ml
```

Then, navigate to:

```
https://<hostname>:8914
```

for local development, `<hostname>` will be `localhost`.


When prompted, enter the development credentials:
 - User: `ssmuser`
 - Password: `ssm2020!`

From there, click on / launch the `ssmdemo.ipynb` notebook!
