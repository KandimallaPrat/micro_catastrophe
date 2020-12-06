import setuptools

with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name = "cat_analysis",
    version = "1.0.0",
    author = "Pratyush Kandimalla (@KandimallaPrat)",
    author_email = "pkandima@caltech.edu",
    description = "Analysing Microtubule Time to Catastrophe",
    long_description = long_description,
    packages = setuptools.find_packages(),
    install_requires = ["numpy", "pandas", "bokeh"],
    classifiers = (
        "Programming Language :: Python :: 3", 
        "Operating System :: OS Independent",
        )
    
    )