from django.shortcuts import render_to_response
from django.template import RequestContext, loader
from django.http import HttpResponse, JsonResponse
import datetime as dt
import json, sys,os, requests, multiprocessing,re
from apiclient.discovery import build
from risaachi.settings import PROJECT_ROOT
from tools import *
import tempfile,shutil,time
from textstat.textstat import textstat
import StringIO
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import nltk,string
from sklearn.cluster import KMeans
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity

STATIC_ROOT = os.path.join(PROJECT_ROOT, 'staticfiles/')
nltk.data.path.append(os.path.join(STATIC_ROOT, 'nltk_data'))
# Key codes we created earlier for the Google CustomSearch API
search_engine_id = '010423994788683144332:j3i7ap0_j94'
api_key = 'AIzaSyCzeSHurkqcumf-NYV7unEBOn0aFKqRs7s'
api_key='AIzaSyAdfzGG90ilBh8FicwJaZh2ZDgNLcRlUMg' #a spare api key to use
pdf_limit=2


stopwords = nltk.corpus.stopwords.words('english')
# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem_and_POS(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    acceptable_pos=['NN','NNS','NNP','NNPS','FW','VB']
    text=text.lower() #lowercase everything
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if token not in stopwords: #remove stopwords
                
                filtered_tokens.append(token)
    #stems = [stemmer.stem(t) for t in filtered_tokens]
    stems=filtered_tokens
    #look at part of speech and only keep nouns, foreign words and verbs
    tagged = nltk.pos_tag(stems)
    POS_tokens=[]
    for tag in tagged:
        text=tag[0]
        pos=tag[1]
        if pos in acceptable_pos:
            if len(text)>2 and "\\" not in text and "\\\\" not in text:
                POS_tokens.append(text.encode('utf-8'))


    return POS_tokens


#this function downloads the urls and saves them locally to be processed
def saveFiles(title,rank,url,sizeLim,ftype,return_dict):
    #return_dict[complexity, text, size, link-ok]
    try:
        req = requests.head(url,verify=False,timeout=1)
        if req.status_code !=200:
            return_dict[rank]=["","","","0"]
            return 1
    except requests.exceptions.RequestException as e:    # This is the correct syntax
        print e
        return_dict[rank]=["","","","0"]
        return 1
    except requests.Timeout as err:
        print e
        return_dict[rank]=["","","","0"]
        return 1
    #we request the header again, just to make sure this server doesnt have too many request restrictions
    try:
        req = requests.head(url,verify=False,timeout=1)
        if req.status_code !=200:
            return_dict[rank]=["","","","0"]
            return 1
    except requests.exceptions.RequestException as e:    # This is the correct syntax
        print e
        return_dict[rank]=["","","","0"]
        return 1
    except requests.Timeout as err:
        print e
        return_dict[rank]=["","","","0"]
        return 1
    #too many requests, give up on this link
    if req.status_code==429:
        print "FAILED: "+title
        return_dict[rank]=["","","","0"]
        return 1
    #check file size first
    if "Content-Length"  in req.headers.keys():
        content_size=req.headers["Content-Length"]
 
    else:
        #some sites dont have content-length in their headers. We ignore these sites
        if ftype not in ['pdf','ppt','doc']:
            #assume that html pages cant be that big anyways..
            content_size=10000
        else:
            content_size=10000000
        
    #file is too large, dont waste time download this
    #print str(content_size)+str(title)
    if int(content_size)>sizeLim:
        #print str(title)+' too large'
        return_dict[rank]=return_dict[rank]=["","",str(float(float(content_size)/1000)),"1"]
        return None

    try:
        req = requests.get(url,verify=False, timeout=3)
        if req.status_code != 200:
            return_dict[rank]=["","","","0"]
            return 1
    except requests.exceptions.RequestException as e:    # This is the correct syntax
        print e
        return_dict[rank]=["","","","0"]
        return 1
    except requests.Timeout as err:
        print e
        return_dict[rank]=["","","","0"]
        return 1
    
    #req = urllib2.urlopen(url)

    #name the file as the current rank and save it
    # with open(fname,"wb") as dfile:
    #     dfile.write(req.content)
    stringio = StringIO.StringIO()
    stringio.write(req.content)
    stringio.seek(0)
    #convert saved file to text
    if ftype=='pdf':
        text=convert_pdf_to_txt(stringio)
    elif ftype=='ppt':
        text=convert_ppt_to_txt(stringio)
    elif ftype=='doc':
        text=convert_doc_to_txt(stringio)
    else:
        text=convert_html_to_txt(stringio,5)
    # with open(rank+".txt","wb") as dfile:
    #     dfile.write(text)    
    
    #f
    #compute text complexity
    try:
        complexity=textstat.flesch_reading_ease(text)
    except:
        complexity=0
    # grade_level=re.findall(r'\d+',complexity)
    
    # complexity=int(grade_level[-1])#get last grade level
    #s=Textatistic(str(text))
    #complexity=s.flesch_score
    # if float(complexity)>100:
    #     complexity=0.0
    # #sometimes it goes negative, set to 0
    # elif float(complexity)<0:
    #     complexity=0.0
    text=unicode(text, 'utf-8')

    text=tokenize_and_stem_and_POS(text)
    return_dict[rank]=[str(complexity),str(text),str(float(float(content_size)/1000)),"1"]

   
    
    return 1


def index(request):
    return render_to_response('index.html')

def rerank(urls,weight,c_score_dict):
    #input is all the urls in order of google's rank
    grank_weight=weight[0]
    complexity_weight=weight[1]
    centrality_weight=weight[2]

  
    #grank is the google rank #
    for grank,url in enumerate(urls):
        
        #only if this url was parsed and makes sure is in the best cluster we looked at
        if 'h' in url:
            complexity=url['h']
     
            
            centrality=c_score_dict[str(grank)]
            #sometimes complexity go over 100, shouldnt happen but it does. Set it to 100
            urls[grank]['i']=str(float(centrality))
            #rank depends on google rank(max score 100) and complexity(max score 100) and centrality
            s = float("{0:.2f}".format(10*(10.0-float(grank))*grank_weight+float(complexity)*complexity_weight+float(centrality)*centrality_weight)) 
            urls[grank]['e']=s
        else:
            #set to 0 if not parsed
            urls[grank]['e']=0.0
    urls = sorted(urls, key=itemgetter('e'), reverse=True) 

    return urls

def googleSearch(request):
    # The build function creates a service object. It takes an API name and API 
    # version as arguments. 
    service = build('customsearch', 'v1', developerKey=api_key)
# A collection is a set of resources. We know this one is called "cse"
# because the CustomSearch API page tells us cse "Returns the cse Resource".
    collection = service.cse()

#deny some file types and websites
    requestGoog = collection.list(q=request.GET['myapp']+" lecture OR notes -video -vimeo -wikipedia -youtube",num=10, start=1,cx=search_engine_id)
    response = requestGoog.execute()
    #output = json.dumps(response, sort_keys=True, indent=2)

    return response
def search(request):
    
    urls=[]
    titles=[]
    OFFLINE_MODE=False
    template = loader.get_template('results.html')
    #different size restriction for different files.
    size={'ppt':2000000,'pdf':1500000,'doc':1500000,'htm':1000000}


    response=googleSearch(request)

 
    #get the final results and display them on the ui
    rank=0
    #temporary directory for this specific search
    pdf_count=0
    jobs=[]
    po=[]
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for i in response['items']:
        tag=get_file_type(i['link'])
        urls.append({'a':i['link'],'b':strip_tags(i['htmlTitle']).encode('utf8'),'c':i['snippet'].encode('utf8'),'d':tag})
        #limit files to 3 mb files
        #queue = Queue() 
        if tag=='pdf' and pdf_count>=pdf_limit:
            pass
        else:
            p=multiprocessing.Process(target=saveFiles,args=(strip_tags(i['htmlTitle']),str(rank), i['link'],size[str(tag)], tag,return_dict))
            p.start()
            jobs.append(p)
            po.append(strip_tags(i['htmlTitle']))
      
            if str(tag)=='pdf':
                pdf_count+=1
        rank+=1
    
    #multiprocess downloading of these files
    # for p in jobs:
    #     p.start()
    #wait for jobs to finish

    for id,p in enumerate(jobs):
        
        p.join()
        print str(po[id].encode('utf8'))+' FINISHED'
    
    

    documents=[]
    documents_keys=[]
    documents_names={}

    #assign the calculated complexity and parsed text to urls to display on ui

    for k in return_dict.keys():
     
        text=return_dict[k][1]
        com=return_dict[k][0]
        linkok=return_dict[k][3]
        fsize=return_dict[k][2]
        if linkok=="1":
            urls[int(k)]['f']="OK"
            if text!="":
                urls[int(k)]['g']=fsize;
                urls[int(k)]['h']=com #complexity
                documents.append(return_dict[k][1]) #text
                documents_keys.append(k)
                documents_names[k]=urls[int(k)]['b']
            else:


                urls[int(k)]['g']="Too Large";
                urls[int(k)]['n']="danger"
        else:
            urls[int(k)]['f']="BAD"
            urls[int(k)]['m']="danger"
    for k in range(0,10):
        if str(k) not in return_dict.keys():
            if urls[int(k)]['d']=="pdf" and pdf_count>=pdf_limit:
                urls[int(k)]['f']="PDF LIM"
                urls[int(k)]['m']="danger"

        
    #filter stop words, tokenize everything and set min_df higher so we are much pickier with terms.
    vect = TfidfVectorizer(min_df=3,ngram_range=(1,2))
    tfidf = vect.fit_transform(documents)


    #get centrality vector

   # centrality_score= (tfidf * tfidf.T).A
    # indices = np.argsort(vect.idf_)[::-1]
    # features = vect.get_feature_names()
    # top_n = 1000
    # top_features = [features[i] for i in indices[:top_n]]
    #print top_features


    #remove the 1 from the centrality score, we dont need it since it's the score with the doc itself
    #centrality_score[centrality_score>0.99]=0

  
    #apply kmeans on two cluster
    # km2 = KMeans(n_clusters=2)
    # km2.fit(tfidf)
    # clusters2=km2.labels_.tolist()
    km = KMeans(n_clusters=2)
    km.fit(tfidf)
    #find which document belongs to which cluster
    clusters=km.labels_.tolist()
    # print "two:"+str(clusters2)
    print documents_names
    print clusters
    print km.cluster_centers_
    #get the largest cluster id
    max_cluster=max(set(clusters), key=clusters.count)
    print 'max: '+str(max_cluster)
    centroid_tfidf=km.cluster_centers_[max_cluster,:] #there are two cluster centroids. We take the centroid of the largest cluster to get the centroid tfidf vector
    
  
    #take out the 0 value we filled since it doesn't do anything. And get the centroid
    #centroid= np.sum(centrality_score,axis=0)/(len(documents)-1)
    #centroid=cluster_centroid
    
    scores=[]
    best_score=0
    best_doc=0
    second_best_doc=0
    #find distance between centroid and document to get a score of the similarity. This naturally bias towards documents in the same cluster. Docs not in the cluster will have to go a greater distance

    for l in range(0,len(documents)):
    
        #centroid is treated like another vector
        news=vstack([tfidf[l],centroid_tfidf])
        #get cosine similarity between this doc and the centroid
        centrality_score= (news * news.T).A
        #s=1 - cosine_similarity(news)
        print "score: "+str(centrality_score[0,1])
        s=centrality_score[0,1]
        #s=np.linalg.norm(centroid-centrality_score[:,l])
        #keep track of best performing doc
        if s>best_score:
            best_score=s
            second_best_doc=best_doc
            best_doc=l
            #print best_doc
        scores.append(s)
    scores=scores/np.linalg.norm(scores)#normalize the score
    c_score_dict={}
  

    # Calculate frequency distribution between the top 2 hits. We assume those two share a lot in common
   
    popular_words=[]

    A=''.join(documents[int(best_doc)])
    B=''.join(documents[int(second_best_doc)])
    fdist = nltk.FreqDist(nltk.word_tokenize(A+B))

    # Output top 10 words
  
    for word, frequency in fdist.most_common(10):
        print('%s;%d' % (word, frequency)).encode('utf-8')
        if len(word)>2 and '\\' not in word:
            if request.GET['myapp'] not in word:  #don't need the original search term
                word=word.translate(None,"'")
                popular_words.append({'a':word.encode('utf-8','ignore')})
  

    #get the centralityt scores and pass it to rerank()
    num=0
    for k in documents_keys:
        if num < len(scores):
            s = float("{0:.2f}".format(100.0*scores[num])) 
        else:
            s=0.0
        c_score_dict[str(k)]=s
        num+=1


    
    #popular words display
    end=popular_words
    urls=rerank(urls,[0.05,0.15,0.8],c_score_dict)
    #update UI with some nice color indicator of top scorers
    biggest_total=0
    biggest_cen=0
    biggest_com=0
    total_id=-1
    cen_id=-1
    com_id=-1
    for id,url in enumerate(urls):
        if 'e' in url:
            total=url['e']
            if float(total)>float(biggest_total):
                biggest_total=total
                total_id=id
        if 'i' in url:
            cen=url['i']
            if float(cen)>float(biggest_cen):
                print str(cen)+"vs"+str(biggest_cen)
                biggest_cen=cen
                cen_id=id
        if 'h' in url:
            com=url['h']
            if float(com)>float(biggest_com):
                biggest_com=com
                com_id=id
    #update the highest value with color
    if total_id>-1:
        urls[total_id]['j']='success'
    if cen_id>-1:
        urls[cen_id]['k']='info'
    if com_id>-1:
        urls[com_id]['l']='warning'
    context = {
        'urls': urls,
        'popular_words': popular_words
    }
  


    return HttpResponse(template.render(context,request))