"""Setup script for the mlfont package."""

import setuptools

VERSION = "0.1.0"

with open("README.md", "r", encoding="utf-8") as file_object:
    LONG_DESCRIPTION = file_object.read()

with open("requirements.txt", "r", encoding="utf-8") as file_object:
    INSTALL_REQUIRES = [
        line.strip()
        for line in file_object.read().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

setuptools.setup(
    name="mlfont",
    version=VERSION,
    author="Quan Wang, Yiran Mao",
    author_email="quanw@google.com",
    description=(
        "Machine-Learning-Guided Range-Compact Web Font Slicing "
        "(Frequency Slicing, Online LDA, DNN Topic Modeling, and Range-Compact Genetic Algorithm)"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/wq2012/mlfont",
    packages=setuptools.find_packages(include=["mlfont", "mlfont.*"]),
    scripts=[
        "scripts/prepare_data.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/export_model.py",
        "scripts/inference.py",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Fonts",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
)
