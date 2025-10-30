#!/usr/bin/env python3
"""
Setup script for Unified LLM Router v2.0
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, filename), encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements(filename):
    requirements = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

setup(
    name='unified-llm-router',
    version='2.0.0',
    description='Unified LLM Router v2.0 - Complete integration of v1.4 (8 components) + v2.0 (10 components)',
    long_description=read_file('BENCHMARK_RESULTS.md') if os.path.exists('BENCHMARK_RESULTS.md') else 'Unified LLM Router v2.0',
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/llm-router-v14',
    license='MIT',

    # Package configuration
    packages=find_packages(include=['src', 'src.*', 'router', 'router.*']),
    python_requires='>=3.10',

    # Dependencies
    install_requires=read_requirements('requirements.txt'),

    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ],
        'api': [
            'fastapi>=0.100.0',
            'uvicorn[standard]>=0.23.0',
        ],
        'gpu': [
            'faiss-gpu>=1.7.4',
        ],
    },

    # Entry points
    entry_points={
        'console_scripts': [
            'unified-router=src.unified.unified_router:main',
            'benchmark-router=benchmark_unified:main',
        ],
    },

    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

    # Additional metadata
    keywords='llm router ai machine-learning model-selection',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/llm-router-v14/issues',
        'Source': 'https://github.com/yourusername/llm-router-v14',
        'Documentation': 'https://github.com/yourusername/llm-router-v14/blob/main/BENCHMARK_RESULTS.md',
    },

    # Include package data
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt', '*.yml', '*.yaml'],
    },

    # Zip safe
    zip_safe=False,
)
