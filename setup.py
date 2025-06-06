#!/usr/bin/env python
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="codeviz",
        version="0.1.0",
        packages=find_packages(),
        include_package_data=True,
        entry_points={
            "console_scripts": [
                "codeviz=codeviz.cli:app",
            ],
        },
    )