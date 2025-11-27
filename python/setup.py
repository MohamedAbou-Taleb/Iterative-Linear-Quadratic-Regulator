from setuptools import setup, find_packages

"""
This is the setup file for 'pip install -e .'
It's the modern, robust way to manage dependencies and package your project.
"""

setup(
    name='class_files',
    version='0.1.0',
    packages=['class_files'],
    
    # Core dependencies required for your project to run
    install_requires=[
        'requests',
        'numpy',
        'jax',
        'matplotlib',
        'vtk',
        'opencv-python',
        'jaxopt',
        'sympy',
        # e.g., 'pandas', 'numpy'
    ],
    
    # Optional dependencies, e.g., for development or specific features
    # Install with: pip install -e .[dev]
    extras_require={
        'dev': [
            'black',     # For code formatting
        ]
    },
)