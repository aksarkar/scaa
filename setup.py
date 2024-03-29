import setuptools

setuptools.setup(
    name='scaa',
    description='Single Cell Adversarial Alignment',
    version='0.1',
    url='https://www.github.com/aksarkar/scaa',
    author='Abhishek Sarkar',
    author_email='aksarkar@uchicago.edu',
    license='MIT',
    install_requires=[
        'h5py',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'torch',
    ],
    packages=setuptools.find_packages(),
)
