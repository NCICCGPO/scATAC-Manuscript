from setuptools import find_packages, setup


setup(
    name='seq2atac',
    version='0.0.1',
    author='Arvind Kumar, Lakssman Sundaram',
    author_email='akumar22@illumina.com',
    description='Seq2ATAC tools',
    long_description="some sequence to atac learning tools",
    long_description_content_type="text/markdown",
    url='https://git.illumina.com/akumar22/seq2atac/',
    license="Closed source. Not for replication",
    packages=find_packages(include=['seq2atac']),
    install_requires=[], # Empty since assuming running inside python38 environment
)