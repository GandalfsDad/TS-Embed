from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="TS-Embed",
    version="0.0.0",
    author="GandalfsDad",
    description="This is a package for the TS-Embed projec",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GandalfsDad/TS=Embed",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11", #To be verified
    install_requires=[
        'torch'
    ],
    packages=find_packages(),
    entry_points={
    }
)
