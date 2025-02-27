from setuptools import find_packages, setup

with open("Readme.md","r") as f:
    long_description = f.read()

setup(
    name="nnanalyzer",
    version="0.1.1",
    description="A series of tools created to analyze neural networks ",
    package_dir={"":"app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sharifware/Large-Scale-Design-and-Analysis-of-Neural-Networks",
    author="ERAU Large Scale Design & Analysis of Neural Networks",
    install_requires=["matplotlib", "torch", "pandas", "numpy", "scikit-learn", "regex",
                      "scipy"],
    python_requires=">3.8",
)