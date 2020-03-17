from setuptools import setup, find_packages

setup(
    name="word_embedding_warehouse_builder",
    version="1.0.0-beta",
    description="Word Embedding warehouse Builder",
    url="https://github.com/shayanfazeli/word_embedding_warehouse_builder",
    author="Shayan Fazeli",
    author_email="shayan@cs.ucla.edu",
    license="Apache",
    classifiers=[
          'Intended Audience :: Science/Research',
          #'Development Status :: 1 - Beta',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
    keywords="machine learning,anxiety,deep learning,inference",
    packages=find_packages(),
    python_requires='>3.6.0',
    scripts=[
        'bin/build_word_embedding_warehouse',
    ],
    install_requires=[
        'numpy>=1.16.4',
        'numpydoc>=0.9.1',
        'torch>=1.3.0',
    ],
    zip_safe=False
)
