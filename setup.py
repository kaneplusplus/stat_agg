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
    install_requires=["sdm", "cnidaria", "collections", "functools", "sklearn"]
    entry_points={
      'console_scripts':[
        'elr=elr:main',
        'clear_env=clear_env:main'
      ],
    },
)
