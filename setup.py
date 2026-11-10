from setuptools import setup, find_packages

setup(
    name='ts_profile_fitter',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'pyyaml',
        'matplotlib',
    ],
    python_requires='>=3.8',
)
