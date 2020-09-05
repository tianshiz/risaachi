import olefile,re,string
from bs4 import BeautifulSoup
from HTMLParser import HTMLParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
import timeit

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)
def convert_pdf_to_txt(path):
  

    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    #laparams.all_texts = True
    laparams.word_margin = float(0.15)
    #laparams=None
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    #fp = file(path, 'rb')
    fp=path
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()

    return text

def convert_ppt_to_txt(path):
    if olefile.isOleFile(path):
        ole=olefile.OleFileIO(path,path_encoding=None)
        a=ole.openstream('PowerPoint Document')
        ret=a.read() #get a string of the stream
    else:
        return ''
        # ret=ret.decode('utf-8',errors='ignore')
        # ret=ret.encode("ascii","ignore")
    ret=re.sub(r"[^\w'\.\,\-\s\d]", '', str(ret))
    return ret.encode('utf8')
def convert_doc_to_txt(path):
    if olefile.isOleFile(path):
        ole=olefile.OleFileIO(path,path_encoding=None)
        a=ole.openstream('WordDocument')
        ret=a.read() #get a string of the stream
    else:
        return ''

    ret=re.sub(r"[^\w'\.\,\-\s\d]", '', str(ret))
    return ret.encode('utf8')

def convert_html_to_txt(path,max_links):
    #source http://stackoverflow.com/questions/22799990/beatifulsoup4-get-text-still-has-javascript
    #default unicode
    path.seek(0)
    html = path.read()

    soup = BeautifulSoup(html,"lxml")
    count=0
    #TODO. IF it has any special characters like in javascript, ignore ;,:,{},
    # for link in soup.findAll('a', href=True):
    #     count+=1
    # if count>25
   # if len(links)>
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text.encode('utf8')


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
def get_file_type(link):

    if '.pdf' in link:
        return 'pdf'
    elif '.PDF' in link:
        return 'pdf'
    elif '.ppt' in link:

        return 'ppt'
    elif '.PPT' in link:
        return 'ppt'
    elif '.doc' in link:

        return 'doc'
    else:
        return 'htm'