#!/usr/bin/env python

"""The setup script."""

import os
from setuptools import setup, find_packages

THIS_DIR = os.path.dirname(__file__)
GITLAB_REPO = 'https://code.ornl.gov/rse/datastreams/ssm/clients'

def read_requirements_from_file(filepath):
    """
    Read a list of requirements from the given file and split into a
    list of strings. It is assumed that the file is a flat
    list with one requirement per line.

    Args:
        filepath:
            Path to the file to read
    Returns:
        A list of strings containing the requirements
    """
    with open(filepath, 'rU') as req_file:
        return req_file.readlines()

setup_args = dict(
    install_requires=read_requirements_from_file(
        os.path.join(
            THIS_DIR,
            'requirements.txt')),
    tests_require=read_requirements_from_file(
        os.path.join(
            THIS_DIR,
            'requirements-dev.txt')))

install_requires = []
tests_requires = []

for line in setup_args['install_requires']:
    if "==" in line:
        install_requires.append(line)
        tests_requires.append(line)
        
for line in setup_args['tests_require']:
    if "==" in line:
        tests_requires.append(line)

setup(
    author="Robert Smith",
    author_email='smithrw@ornl.gov',
    python_requires='>=3.6',
    include_package_data=True,
    name='ssm_ml',
    packages=find_packages(),
    py_modules=['Filter', 'ssmml'],
        #include=['ssm_ml']),
    setup_requires=['pytest-runner'],
    install_requires=install_requires,
    tests_require=['nose'],
    test_suite='nose.collector',
    version='0.0.13',
    zip_safe=False
)
