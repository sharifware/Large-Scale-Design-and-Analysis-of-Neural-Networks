from setuptools import find_packages, setup

with open("README.md","r") as f:
    long_description = f.read()

setup(
    name="nnanalyzer",
    version="0.1.0",
    description="A series of tools created to analyze neural networks ",
    package_dir={"":"src"},
    packages=find_packages(where="ap"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sharifware/Large-Scale-Design-and-Analysis-of-Neural-Networks"
    author="ERAU Large Scale Design & Analysis of Neural Networks",
    install_requires=["matplotlib", "os", "importlib.util", "torch" "pandas", "numpy", "sklearn", "standardRegArchitercture",
                      "scipy"],
    python_requires=">3.8",
)