from setuptools import find_packages
import setuptools

setuptools.setup(
    name = "conversions_and_maths",
    version = "0.0.1",
    author = "Fahad_Abul",
    author_email = "fahadsohailabul@gmail.com",
    description = "An example  package to perform unit conversions and do maths stuff",
    packages = ['maths', 'conversions'],
)