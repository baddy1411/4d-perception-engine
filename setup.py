from setuptools import setup, find_packages

setup(
    name="4d_perception_engine",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyspark",
        "torch",
        "numpy",
        "fiftyone",
        "boto3",
    ],
)
