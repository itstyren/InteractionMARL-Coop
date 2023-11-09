from setuptools import setup, find_packages


setup(
    name='Interaction-SD',
    version='0.1',
    author='tyren',    
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'setproctitle',
        'wandb',
        'pettingzoo',
        'tensorboardX',
        'stable_baselines3',
        'imageio',
        'torchinfo',

    ],
)