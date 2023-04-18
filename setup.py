import os
from setuptools import setup, find_packages


with open(os.path.join("src", "backprompt", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__ = "):
            version = str(line.split()[-1].strip('"'))
            break


requirements_base = [
    "torch>=1.12.1",
    "transformers>=4.26.1",
]

requirements_demos = [
    "jupyter>=1.0.0",
    "pandas>=1.5.3",
]

requirements_dev = [
    "black>=23.1.0",
    "docutils<0.19",
    "pydata-sphinx-theme>=0.13.1",
    "pytest>=7.2.1",
    "pytest-cov>=4.0.0",
    "scikit-learn>=1.2.2",
    "sphinx>=6.1.3",
    "sphinx-togglebutton>=0.3.2",
    "sphinxcontrib-napoleon>=0.7",
    "twine>=4.0.2",
]


with open("README.md", mode="r", encoding="utf-8") as f:
    readme = f.read()


setup(
    name="backprompt",
    version=version,
    description="backprompt is to prompt engineering as micrograd is to deep learning",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/kddubey/backprompt/",
    license="Apache License 2.0",
    python_requires=">=3.8.0",
    install_requires=requirements_base,
    extras_require={
        "demos": requirements_base + requirements_demos,
        "dev": requirements_base + requirements_demos + requirements_dev,
    },
    author_email="kushdubey63@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
