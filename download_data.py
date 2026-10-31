"""Optional: only download the MovieLens data:   python download_data.py"""
from src.data import download

if __name__ == "__main__":
    print("Data ready in", download())
