"""Entry point for preprocessing (v2). See src/preprocessing_v2.py for the pipeline."""
from src.preprocessing import main

if __name__ == "__main__":  # required on Windows for multiprocessing
    main()