try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='stat_agg',
    version='0.2',
    author='Michael J. Kane',
    author_email='kaneplusplus@gmail.com',
    packages=['stat_agg'],
    license='LICENSE.txt',
    description='Ensemble learning',
    long_description=open('README.md').read(),
    install_requires=["scikit-learn >= 0.17", "pandas >= 0.17.1"]
)
