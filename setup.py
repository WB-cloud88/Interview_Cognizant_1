from setuptools import setup, find_packages

setup(
    name="transformer_failure_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "mlflow>=2.8.0",
        "shap>=0.42.0",
        "pyyaml>=6.0",
    ],
)
