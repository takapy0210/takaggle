import os
from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='takaggle',
    version='1.0.1',
    description='A set of scripts used in the data analysis competition',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Takanobu Nozawa',
    author_email='takanobu.030210@gmail.com',
    url='https://github.com/takapy0210/takaggle',
    license='MIT License',
    install_requires=read_requirements(),
    packages=find_packages(exclude=('tests')),
    python_requires='~=3.6'
)