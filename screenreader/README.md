## Screen Reader
# Introduction
Screen reader is simple python script that makes a screenshot, infers what text is present in the screenshot, and outputs text inferences made to a .txt file. To take the screenshot the pyscreenshot library was utilized and to make text inferences the tesseract library was used. 
# Applications
This tool was originally created with the intention of speeding up the process of copying text from an image or pdf that does not allow copying directly through `⌘C` or `Ctrl+C`
# Pre-requisites
* **Python 2.6+ (only tested with python 2.7) to download python visit: https://www.python.org/downloads/**
* **Libraries**
  - Tesseract
    - `pip install Pillow`
    - `pip install -U git+https://github.com/tesseract-ocr/tesseract`
    - `pip install pytesseract` alternative discussed here: https://pypi.org/project/pytesseract/
  - Pyscreenshot
    - `pip install pyscreenshot`
# Getting started
Clone the project
```git clone https://github.com/Benthijs/screenreader```
# Example usage
To use this simple tool, open terminal, cd into the directory of `screenreader.py` and type:
```cd /path/to/screenreader``` (the directory encompassing the program)
Then run the following:
```python screenreader.py```
Whereafter you should immediately switch to the tab of the text that needs to be interpreted as a screenshot will be taken.
# Getting started
Clone the project
```git clone https://github.com/Benthijs/screenreader```
# Issues
* **Inference accuracy is not great**
# Future goals
* **Improve user interface**
* **Depricate dependancy on other libraries**
* **Implement a more advanced markov chain with the intention of improving text quality**
# License
None currently as this is a very simple amalgamation of others work.
## Supporting Documentation
https://github.com/tesseract-ocr/tesseract/wiki/Documentation

https://pyscreenshot.readthedocs.io/en/latest/_modules/pyscreenshot.html

https://pillow.readthedocs.io/en/5.3.x/
