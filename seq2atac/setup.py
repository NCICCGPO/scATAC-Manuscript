from setuptools import find_packages, setup

setup(
    name='seq2atac',
    version='0.0.1',
    author='Arvind Kumar, Lakssman Sundaram',
    description='Seq2ATAC tools',
    long_description="some sequence to atac learning tools",
    long_description_content_type="text/markdown",
    license="Closed source. Not for replication",
    packages=find_packages(include=['seq2atac']),
    install_requires=[], 
)