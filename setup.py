from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="khadee-eda",
    version="1.0.1",
    author="Khadee",
    description="Deep Insights EDA — Comprehensive data profiling with global AI techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khadee/khadee-eda",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "khadee_eda": [
            "assets/*.css",
            "assets/*.js",
            "renderers/templates/*.html",
        ],
    },
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "plotly>=5.10.0",
        "openpyxl>=3.0.0",
        "pyarrow>=10.0.0",
        "jinja2>=3.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="eda, data-profiling, exploratory-data-analysis, data-science, machine-learning",
)
