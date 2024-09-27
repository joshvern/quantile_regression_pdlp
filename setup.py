# setup.py

from setuptools import setup, find_packages

setup(
    name='quantile_regression_pdlp',
    version='0.1.0',
    author='Joshua Vernazza',
    author_email='vernazzajosh@gmail.com',
    description='Quantile Regression using PDLP solver from Google\'s OR-Tools',
    url='https://github.com/joshvern/quantile_regression_pdlp', 
    packages=find_packages(),
    install_requires=[
        'ortools>=9.4',
        'numpy>=1.17.0',
        'pandas>=1.0.0',
        'scipy>=1.4.0',
        'tqdm>=4.0.0'
    ],
    python_requires='>=3.6',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
)
