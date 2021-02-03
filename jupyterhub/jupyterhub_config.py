# Configuration file for jupyterhub.

c.JupyterHub.spawner_class = 'simple'
c.Spawner.notebook_dir = '/srv/jupyterhub/notebooks'
c.Spawner.args = ['--allow-root']

