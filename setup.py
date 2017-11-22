from setuptools import setup, find_packages

setup(name='deeptesla',
	version='1.0',
	packages=find_packages(),
	discription='DeepTesla Dataset',
	author='ensc424',
	licence='MIT',
	install_requires=[
		'keras',
		'h5py'
	],
	zip_safe=False)