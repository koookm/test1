"""
Setup configuration for Expert Investment Team package.
"""

from setuptools import setup, find_packages

setup(
    name="expert_investment_team",
    version="1.0.0",
    description=(
        "Multi-Agent LLM Investment System based on arXiv:2602.23330 "
        "'Toward Expert Investment Teams: A Multi-Agent LLM System with Fine-Grained Trading Tasks'"
    ),
    author="Research Implementation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "anthropic>=0.40.0",
        "openai>=1.50.0",
        "yfinance>=0.2.40",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "requests>=2.31.0",
        "scipy>=1.11.0",
        "loguru>=0.7.0",
        "tenacity>=8.2.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.8.0", "seaborn>=0.13.0", "plotly>=5.18.0"],
        "opt": ["cvxpy>=1.4.0", "pypfopt>=1.5.5"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "full": [
            "matplotlib>=3.8.0", "seaborn>=0.13.0", "plotly>=5.18.0",
            "cvxpy>=1.4.0", "pypfopt>=1.5.5",
            "pytest>=7.4.0", "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "investment-team=main:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Office/Business :: Financial :: Investment",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
    ],
)
