import shutil
import os
import subprocess
import PyPDF2
#from pdfminer.high_level import extract_text
# import tempfile

# temp_dir = tempfile.TemporaryDirectory()
# temp_dir_path = temp_dir.name

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
    if extension.lower() == ".pdf":
        lst = []
        print("Extension is already ", extension.lower())
        #pdfFileObj = open(os.path.join(directory, filename), 'rb')
        pdfFileObj = open(doc_path, 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]
            page = pageObj.extract_text()
            lst.append(page)
        txt = ''.join(lst)
        # closing the pdf file object
        pdfFileObj.close()
        #txt = subprocess.check_output(["pdftotext", doc_path, "-"]).decode()
        #print(txt[:200])
    else:
        subprocess.call(["libreoffice", "--headless", "--convert-to", "pdf", doc_path, "--outdir", destination])
        #txt = extract_text(os.path.join(destination, new_name))
        txt = subprocess.check_output(["pdftotext", os.path.join(destination, new_name), "-"]).decode()
    
    shutil.rmtree(destination)
    return txt
