# Configuration file for jupyterhub.

c.JupyterHub.base_url = '/machine-learning/training/'

c.JupyterHub.spawner_class = 'simple'
c.Spawner.notebook_dir = '/srv/jupyterhub/notebooks'
c.Spawner.args = ['--allow-root']

