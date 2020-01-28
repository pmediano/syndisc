from setuptools import setup, find_packages

setup(name="syndisc",
        packages=find_packages(),
        provides=["syndisc"],
        version="0.1",
        author="Pedro Mediano and Fernando Rosas",
        url="https://www.information-dynamics.org",
        install_requires=[
            "dit",
            "pypoman",
            "cvxopt",
            "numpy",
            "scipy",
            "networkx",
            "matplotlib",   # Actually required by pypoman
            ],
        long_description=open("README.md").read(),
        license="BSD",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3.3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Physics",
            ],
        )

