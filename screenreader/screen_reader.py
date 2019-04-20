import os
import time
import pytesseract
from PIL import Image
import pyscreenshot as capture
dir = 'img.png'
begin = time.time()
while(True):
    if((time.time() - begin) >= 3): # add three second timer
        capture.grab().save(dir)
        r = pytesseract.image_to_string(Image.open(dir).convert('LA'), lang='eng')
        # convert image to gray-scale and then to text such that the model can more accurately make inferences
        os.remove(dir)
        with open('out.txt', 'w') as f:
            f.write(r.encode('utf-8'))
        break
