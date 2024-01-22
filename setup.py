from setuptools import setup, find_packages


setup(
    name='InteractionMARL-Coop',
    version='0.1',
    author='Anonymous',    
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