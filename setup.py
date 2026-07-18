"""
Setup configuration for Computer Vision Image Classifier
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cv-image-classifier",
    version="1.0.0",
    author="BONDI DHANUSH",
    author_email="bondidhanush01@gmail.com",
    description="A comprehensive computer vision project for image classification, object detection, and face recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bondidhanush01-bit/computer-vision-image-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "isort>=5.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "cv-classifier=src.train:main",
        ],
    },
    keywords="computer-vision image-classification object-detection face-recognition deep-learning pytorch",
    project_urls={
        "Bug Reports": "https://github.com/bondidhanush01-bit/computer-vision-image-classifier/issues",
        "Source": "https://github.com/bondidhanush01-bit/computer-vision-image-classifier",
        "Documentation": "https://github.com/bondidhanush01-bit/computer-vision-image-classifier#readme",
    },
)
