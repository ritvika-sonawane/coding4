import soundfile as sf
import os

path = "dump/raw/org/train_bilingual/data/format.1/data_wav/103-1240-0000.flac"
if os.path.exists(path):
    print(f"File exists: {path}")
    try:
        data, sr = sf.read(path)
        print(f"Read successful. Shape: {data.shape}, SR: {sr}")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"File not found: {path}")
