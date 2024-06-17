from setuptools import setup

setup(
    name="vlmc_envs",
    version="0.1",
    packages=["envs"],
    install_requires=[
        "gymnasium",
        "numpy",
    ],
)
