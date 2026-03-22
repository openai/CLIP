import os
from setuptools import setup, find_packages

setup(
    name="clip",
    py_modules=["clip"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        stripped
        for line in open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        for stripped in [line.strip()]
        if stripped and not stripped.startswith("#")
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
