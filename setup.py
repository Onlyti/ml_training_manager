from setuptools import setup, find_packages

setup(
    name="ml_training_manager",
    version="0.1.0",
    description="ML 모델 학습 자동화 관리자",
    author="ML Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "wandb>=0.15.0",
        "psutil>=5.8.0",
        "windows-curses;platform_system=='Windows'",
    ],
    entry_points={
        "console_scripts": [
            "ml-training-manager=ml_training_manager.training_manager.main_training_manager:main",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 