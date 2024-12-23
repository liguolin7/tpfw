from setuptools import setup, find_packages

setup(
    name="tfpw",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0.0',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
) 