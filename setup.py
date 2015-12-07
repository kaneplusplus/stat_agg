try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='statagg',
    version='0.2',
    author='Michael J. Kane',
    author_email='kaneplusplus@gmail.com',
    packages=['statagg'],
    license='LICENSE.txt',
    description='Statistical Aggregators for Python',
    long_description=open('README.md').read(),
    install_requires=["scikit-learn >= 0.17", "pandas >= 0.17.1"]
)
