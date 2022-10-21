from setuptools import setup, find_packages

setup(
    name="QuantumDeepField_mol",
    version="0.1.0",
    description="Machine Learning model to predict molecular properties.",
    author="",
    author_email="",
    keywords="python machine-learning deep-learning",
    license="MIT",
    packages=find_packages(),
    install_requires=["tqdm>=0.9.0"
                      "certifi==2022.9.24",
                      "charset-normalizer==2.1.1",
                      "idna==3.4",
                      "numpy==1.23.3",
                      "Pillow==9.2.0",
                      "requests==2.28.1",
                      "scipy==1.9.1",
                      "torch==1.12.1",
                      "torchaudio==0.12.1",
                      "torchvision==0.13.1",
                      "typing_extensions==4.3.0",
                      "urllib3==1.26.12"],
)
