import requests
from jobtastic import JobtasticTask
from tools import *


class tasks(JobtasticTask):
	'''simple download'''
	def saveFiles(rank,url,sizeLim,ftype):

    req = requests.head(url)
    
    #check file size first
    if "Content-Length"  in req.headers.keys():
        content_size=req.headers["Content-Length"]
        print content_size
    else:
        #some sites dont have content-length in their headers. We ignore these sites
        content_size=10000000
        
    #file is too large, dont waste time download this
    if int(content_size)>sizeLim:
        
        return None

    req = requests.get(url)
    self.update_progress(1, 3)
    fname=rank+"."+ftype
    #name the file as the current rank and save it
    with open(fname,"wb") as dfile:
        dfile.write(req.content)
    #convert saved file to text
    if ftype=='pdf':
        text=convert_pdf_to_txt(fname)
    elif ftype=='ppt':
        text=convert_ppt_to_txt(fname)
    elif ftype=='doc':
        text=convert_doc_to_txt(fname)
        
    else:
        text=convert_html_to_txt(fname)
    self.update_progress(2, 3)
    with open(rank+".txt","wb") as dfile:
            dfile.write(text)    
    #remove the downloaded file
    os.remove(fname)
    #compute text complexity
    #TODO
    #return complexity
    
    return 1