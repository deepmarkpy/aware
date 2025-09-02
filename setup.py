from setuptools import setup, find_packages

setup(
    name="AWARE",
    version="0.1.0",
    description="Audio watermarking with small (Δ) adversarial perturbations.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9,<3.13",
    install_requires=[
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "numpy==1.26.4",
        "librosa==0.9.2",
        "soundfile==0.12.1",
        "pydantic==2.5.0",
        "matplotlib==3.7.2",
        "scikit-learn==1.5.0",
        "numba==0.59.0",    
        "resampy==0.4.2",
        "tqdm==4.66.1",
        "pesq==0.0.4",
        "pyyaml==6.0.1",
        "pystoi==0.4.1",
        "webrtcvad==2.0.10"
    ]
)
