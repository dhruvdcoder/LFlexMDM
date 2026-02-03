from setuptools import setup, find_packages

setup(
    name="lflexmdm",
    version="0.1.0",
    description="FlexMDM",
    packages=find_packages(),
    install_requires=[
        "xlm",  # Core XLM framework dependency
        "torch",
        "jaxtyping",
    ],
    package_data={
        "lflexmdm": ["configs/**/*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.11",
)
