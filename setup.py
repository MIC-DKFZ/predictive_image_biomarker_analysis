import setuptools

# with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="predimgbmanalysis",
    author="Division of Medical Image Computing (MIC)",
    author_email="s.xiao@dkfz-heidelberg.de",
    description="Enhancing Predictive Imaging Biomarker Discovery through Treatment Effect Analysis",
    url="https://github.com/MIC-DKFZ/predictive_image_biomarker_analysis",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    license='Apache License Version 2.0, January 2004',
)
