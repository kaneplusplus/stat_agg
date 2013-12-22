try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='stat_agg',
    version='0.1',
    author='Michael J. Kane',
    author_email='kaneplusplus@gmail.com',
    packages=['stat_agg'],
    license='LICENSE.txt',
    description='Ensemble learning with redis',
    long_description=open('README.md').read(),
    install_requires=["py-sdm", "cnidaria", "scikit-learn >= 0.13"],
    entry_points={
      'console_scripts':[
        'stat_agg_bench=stat_agg.stat_agg_bench:main',
        'clear_env=stat_agg.clear_env:main'
      ],
    },
)
