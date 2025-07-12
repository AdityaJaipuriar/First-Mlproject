#It turns the ML project into a Python package, so others (or future me) can install it with pip install and use it like any other library.
from setuptools import find_packages,setup
from typing import List

minus_e_dot = "-e ."
# -e . in requirements.txt because - It tells pip to install the package in editable mode, meaning changes to the source code will immediately reflect without needing to reinstall.

def get_requirements(file_path:str)-> List[str]:
    '''This function will return a list of packages needed to downloaded'''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if(minus_e_dot in requirements):
            requirements.remove(minus_e_dot)

    return requirements
setup(
    name='First-MLproject',
    version = '0.0.1',
    author='Aditya Jaipuriar',
    author_email = 'adityajaipuriar30@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)