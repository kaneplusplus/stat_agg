try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='elr',
    version='0.1',
    author='Michael J. Kane',
    author_email='kaneplusplus@gmail.com',
    packages=['elr'],
    license='LICENSE.txt',
    description='Ensemble learning with redis',
    long_description=open('README.txt').read(),
    install_requires=["py-sdm", "cnidaria", "scikit-learn >= 0.13"],
    entry_points={
      'console_scripts':[
        'elr_bench=elr.elr_bench:main',
        'clear_env=elr.clear_env:main'
      ],
    },
)
