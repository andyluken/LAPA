from setuptools import setup, find_packages

setup(
    name="laq_ad",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "torchvision",
        "accelerate",
        "einops",
        "numpy<2.0.0",
        "Pillow",
        "tqdm",
        "wandb",
        "opencv-python>=4.9.0.80",
        "nuscenes-devkit",
        "pyquaternion",
        "matplotlib"
    ],
)
