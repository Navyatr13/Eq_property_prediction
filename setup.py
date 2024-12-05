from setuptools import setup, find_packages

setup(
    name='EqPropertyPrediction',
    version='0.1.0',
    description='A package for property prediction using GNNs',
    author='Navya Rammesh',
    packages=find_packages(include=['Eqproperty_prediction','Eqproperty_prediction.*']),  # Automatically find and include all packages
    install_requires=[
        'torch',  # PyTorch for ML
        'torch_geometric',  # PyTorch Geometric for graph processing
        'numpy',  # Numerical operations
        'pandas',  # For reading and processing data
    ],
)
