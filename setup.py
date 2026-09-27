from setuptools import setup, find_packages

setup(
    name='ts_profile_fitter',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'pyyaml',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'run-ts-profile-fitter=scripts.run_full_database:main',
        ],
    },
)
