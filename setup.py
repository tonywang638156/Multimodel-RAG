from setuptools import setup, find_packages

setup(
    name="multimodel_rag",
    version="0.1.0",
    description="A video and audio multimodal RAG system using BridgeTower, LanceDB, and Streamlit",
    author="Tony Wang",
    packages=find_packages(where="scripts"),
    package_dir={"": "scripts"},
    include_package_data=True,
    install_requires=[
        # Core utilities
        "numpy>=1.23",
        "pillow>=10.0",
        "tqdm>=4.64",
        "pathlib",
        "scikit-image",
        "matplotlib",
        "pandas",
        "streamlit",

        # Deep learning and transformers
        "torch>=2.0",
        "transformers>=4.39",

        # Video and image processing
        "opencv-python>=4.8",
        "moviepy>=2.1.1",
        "imageio>=2.5",
        "imageio-ffmpeg>=0.6",
        "proglog>=0.1.10",

        # RAG & multimodal
        "lancedb>=0.21.2",
        "datasets>=2.18.0",
        "sentence-transformers>=2.2.2",

        # Utilities
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "base58>=2.1.1",

        # OpenAI
        "openai>=1.0.0",
    ],
    python_requires="==3.13.3",
)
