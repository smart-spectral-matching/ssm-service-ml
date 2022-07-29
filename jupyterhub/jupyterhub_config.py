# Configuration file for jupyterhub.

c.JupyterHub.base_url = '/machine-learning/notebooks/'

c.JupyterHub.spawner_class = 'simple'
c.Spawner.notebook_dir = '/srv/jupyterhub/notebooks'
c.Spawner.args = ['--allow-root']

# NOTE: This must match docker-entrypoint.sh under JPYTR_GROUP
c.LocalAuthenticator.group_whitelist = ['ssm']