from setuptools import setup, find_packages

setup(
    name="mtraining",  # Name of your project
    version="0.1.0",
    packages=find_packages(),  # Automatically discover all packages
    install_requires=[],  # List dependencies if any (or use requirements.txt)
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Specify the Python version
)
