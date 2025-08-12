"""
Setup script for ImageNet-lightweight package.
Allows installation via pip for use as a library.
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="imagenet-lightweight",
    version="1.0.0",
    author="",
    author_email="",
    description="One-GPU benchmarking suite for lightweight vision models on ImageNet subsets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sabdulmajid/ImageNet-lightweight",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "fvcore>=0.1.5",
        "pynvml>=11.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "imagenet-lightweight-train=train:main",
            "imagenet-lightweight-eval=eval:main",
            "imagenet-lightweight-benchmark=benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "docs/*.md"],
    },
    keywords=[
        "imagenet",
        "image classification",
        "deep learning",
        "computer vision",
        "pytorch",
        "benchmarking",
        "lightweight models",
        "mobile networks",
        "efficient neural networks",
    ],
    project_urls={
        "Bug Reports": "https://github.com/sabdulmajid/ImageNet-lightweight/issues",
        "Source": "https://github.com/sabdulmajid/ImageNet-lightweight",
        "Documentation": "https://github.com/sabdulmajid/ImageNet-lightweight/tree/main/docs",
    },
)
