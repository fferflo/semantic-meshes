#!/usr/bin/env python3

from setuptools import setup

setup(
    name="semantic-meshes",
    version="0.1",
    description="Label fusion for semantic meshes",
    author="Florian Fervers",
    author_email="florian.fervers@gmail.com",
    packages=["semantic_meshes"],
    package_data={"semantic_meshes": ["*.so"]},
    scripts=["scripts/colorize_cityscapes_mesh.py", "scripts/colorize_mesh.py"],
    license="MIT",
    install_requires=[
          "tf-semseg",
          "imageio",
          "numpy",
          "tqdm",
      ],
)
