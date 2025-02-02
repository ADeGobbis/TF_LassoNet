from setuptools import find_packages, setup

setup(
    name='tf_lassonet',
    packages=find_packages(),
    version='0.1.0',
    description='LasssoNet in TensorFlow',
    author='Andrea De Gobbis, Luciano Lorenti',
    install_requires=[
        'tensorflow', 
        'tensornetwork',
    ],
    license='MIT',    
    include_package_data=True,
)
