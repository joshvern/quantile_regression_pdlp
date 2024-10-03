# setup.py

from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent.resolve()

# Read the README file for the long description
README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="quantile_regression_pdlp",
    version="0.1.0",
    description=(
        "A Python package for performing quantile regression using the PDLP solver "
        "from Google's OR-Tools, with sklearn and pandas compliance, multi-output support, "
        "weighted regression, and L1 regularization."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/joshvern/quantile_regression_pdlp",
    author="Joshua Vernazza",
    author_email="vernazzajosh@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords="quantile-regression linear-programming machine-learning statistics",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    python_requires=">=3.6",
    install_requires=[
        "ortools>=9.0.9047",
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scipy>=1.4.0",
        "tqdm>=4.50.0",
        "joblib>=1.0.0",
        "scikit-learn>=0.22.0",
    ],
    project_urls={
        "Bug Reports": "https://github.com/joshvern/quantile_regression_pdlp/issues",  # Replace with your issue tracker URL
        "Source": "https://github.com/joshvern/quantile_regression_pdlp/",  # Replace with your repository URL
    },
    include_package_data=True,
)
