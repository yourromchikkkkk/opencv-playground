import os
from pathlib import Path

import pytesseract
from PIL import Image

IMAGE_DIR = Path(__file__).resolve().parent / 'data'
IMAGE_PATH = IMAGE_DIR / 'bluebonnet.png'
text = pytesseract.image_to_string(Image.open(IMAGE_PATH), lang='eng')
print(text)
