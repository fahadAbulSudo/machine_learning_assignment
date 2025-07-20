import shutil
import os
import subprocess
from pdfminer.high_level import extract_text

def convertAndParse(filename, doc_path):
    # create a temporary directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    destination = os.path.join(base_path, "tempPDF")

    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.mkdir(destination)
  
    # convert to pdf and save
    new_name, extension = os.path.splitext(filename)
    new_name +=".pdf"

    # parse the file using pdf miner
    if extension.lower == ".pdf":
        txt = extract_text(doc_path)
    else:
        subprocess.call(["libreoffice", "--headless", "--convert-to", "pdf", doc_path, "--outdir", destination])
        txt = extract_text(os.path.join(destination, new_name))
    
    shutil.rmtree(destination)
    return txt 
