# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:31:54 2019

@author: Kedar Kulkarni
"""

import pandas as pd
import math
import numpy as np
import glob
import datetime as dt
import copy
import sys

def readFile(path, dict):
  count=0
  files = glob.glob(path)
  for name in files:
   dict += open(name,encoding="utf8",errors='ignore').read()
   count+=1
  return dict,count

def findAccuracyNB(path,type,hamDict,spamDict,totalhamcount,totalspamcount,ul,hc,sc):
  files=glob.glob(path)
  right=0
  total=0
  for name in files:
    ph=math.log(hc/(hc+sc),2)
    ps=math.log(sc/(hc+sc),2)
    total+=1
    test=open(name,encoding="utf8",errors='ignore').read().split()
    for word in test:
      if word in ul:
        count = hamDict.count(word)
        ph+=math.log((count+1)/totalhamcount,2)
        count = spamDict.count(word)
        ps+=math.log((count+1)/totalspamcount,2)
    if(ps>ph and type=='spam'):
      right+=1
    elif(ph>ps and type=='ham'):
      right+=1
  return right*100/total

def getDF(df,path,type,method,ul):
  files = glob.glob(path)
  row=len(df)-1
  for name in files:
    row +=1
    df.loc[row,:]=0
    if type=='spam' and method == 'LRTrain':
      df.loc[row,'outputy']=1
    file = open(name,encoding="utf8",errors='ignore').read()
    file=file.split()
    for word in file:
      if word in ul:
        df.loc[row,word]=df.loc[row,word]+1
  return df

t1=dt.datetime.now()

#trainhampath = 'E:/mlass2/train/ham/*.txt'
trainhampath = sys.argv[1]
hamDict,hc=readFile(trainhampath,'')
#hamDict=re.sub('[^A-Za-z]', ' ', hamDict)

#trainspampath = 'E:/mlass2/train/spam/*.txt'
trainspampath = sys.argv[2]
spamDict,sc=readFile(trainspampath,'')
#spamDict=re.sub('[^A-Za-z]', ' ', spamDict)

hl=hamDict.split()
sl=spamDict.split()
kl = hl+sl
ul=pd.DataFrame({'col1':kl})['col1'].unique()
uc=len(ul)

totalhamcount=len(hl)+uc
totalspamcount=len(sl)+uc

print('Naive Baye\'s Approach')
#testspampath="E:/mlass2/test/spam/*.txt"
testspampath = sys.argv[3]
print('spam',findAccuracyNB(testspampath,'spam',hl,sl,totalhamcount,totalspamcount,ul,hc,sc),'%')

#testhampath="E:/mlass2/test/ham/*.txt"
testhampath = sys.argv[4]
print('ham',findAccuracyNB(testhampath,'ham',hl,sl,totalhamcount,totalspamcount,ul,hc,sc),'%')
t2=dt.datetime.now()
print('time',t2-t1)
print()

print('Logistic Regression')

df = pd.DataFrame(columns=np.append(ul,'outputy'))
df = getDF(df,trainspampath,'spam','LRTrain',ul)

df = getDF(df,trainhampath,'ham','LRTrain',ul)
new_col=[1]*len(df)
df.insert(loc=0, column='x0', value=new_col)

wl=[1]*(len(df.columns)-1)
na = np.array(df)
nwa = np.array(wl)
mu=0.1
lamda=0.08
iterations=100
k=na[:,-1]
nna=na[:,:-1]

for rr in range(0,iterations):
  nwl=[]
  i=-1
  o = np.dot(nna,np.array(wl))
  for wi in wl:
    i+=1
    y=k-(1-(1/(1+(np.exp(o)))))
    ssu = np.array((nna[:,i]))*y
    wwi= wi+ ((mu*sum(ssu))-(mu*lamda*wi))
    nwl.append(wwi)
  wl=copy.deepcopy(nwl)

t3=dt.datetime.now()
print('training time',t3-t2)

df1 = pd.DataFrame(columns=ul)
df1 = getDF(df1,testspampath,'spam','LRTest',ul)
lengthcol = [1]*len(df1)
df1.insert(loc=0, column='x0', value=lengthcol)
na1 = np.array(df1)
out = np.dot(na1,np.array(wl))
test=100*(len(out[out>0]))/len(out)
print('spam',test,'%')

df1 = pd.DataFrame(columns=ul)
df1 = getDF(df1,testhampath,'ham','LRTest',ul)
lengthcol = [1]*len(df1)
df1.insert(loc=0, column='x0', value=lengthcol)
na1 = np.array(df1)
out = np.dot(na1,np.array(wl))
test=100*(len(out[out<0]))/len(out)
print('ham',test,'%')

t4=dt.datetime.now()
print('time',t4-t3)

#Filtering
stopwords = ['a','about','above','after','again','against','all','am','an','and','any','are','aren\'t','as','at','be','because','been','before','being','below','between','both','but','by','can\'t','cannot','could','couldn\'t','did','didn\'t','do','does','doesn\'t','doing','don\'t','down','during','each','few','for','from','further','had','hadn\'t','has','hasn\'t','have','haven\'t','having','he','he\'d','he\'ll','he\'s','her','here','here\'s','hers','herself','him','himself','his','how','how\'s','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into','is','isn\'t','it','it\'s','its','itself','let\'s','me','more','most','mustn\'t','my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own','same','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','so','some','such','than','that','that\'s','the','their','theirs','them','themselves','then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','those','through','to','too','under','until','up','very','was','wasn\'t','we','we\'d','we\'ll','we\'re','we\'ve','were','weren\'t','what','what\'s','when','when\'s','where','where\'s','which','while','who','who\'s','whom','why','why\'s','with','won\'t','would','wouldn\'t','you','you\'d','you\'ll','you\'re','you\'ve','your','yours','yourself','yourselves']

nhl=[]
for w in hl:
  if w not in stopwords:
    nhl.append(w)
nsl=[]
for w in sl:
  if w not in stopwords:
    nsl.append(w)

nkl = nhl+nsl
nul=pd.DataFrame({'col1':nkl})['col1'].unique()
nuc=len(nul)

ntotalhamcount=len(nhl)+nuc
ntotalspamcount=len(nsl)+nuc

print()
print('----------------After filtering stopwords--------------------')
print()
print('Naive Baye\'s Approach')
print('spam',findAccuracyNB(testspampath,'spam',nhl,nsl,ntotalhamcount,ntotalspamcount,nul,hc,sc),'%')

print('ham',findAccuracyNB(testhampath,'ham',nhl,nsl,ntotalhamcount,ntotalspamcount,nul,hc,sc),'%')
t5=dt.datetime.now()
print('time',t5-t4)

print()
print('Logistic Regression')

df = pd.DataFrame(columns=np.append(nul,'outputy'))
df = getDF(df,trainspampath,'spam','LRTrain',nul)

df = getDF(df,trainhampath,'ham','LRTrain',nul)
new_col=[1]*len(df)
df.insert(loc=0, column='x0', value=new_col)

wl=[1]*(len(df.columns)-1)
na = np.array(df)
nwa = np.array(wl)
k=na[:,-1]
nna=na[:,:-1]


for rr in range(0,iterations):
  nwl=[]
  i=-1
  o = np.dot(nna,np.array(wl))
  for wi in wl:
    i+=1
    y=k-(1-(1/(1+(np.exp(o)))))
    ssu = np.array((nna[:,i]))*y
    wwi= wi+ ((mu*sum(ssu))-(mu*lamda*wi))
    nwl.append(wwi)
  wl=copy.deepcopy(nwl)

t6=dt.datetime.now()
print('training time',t6-t5)

df1 = pd.DataFrame(columns=nul)
df1 = getDF(df1,testspampath,'spam','LRTest',nul)
lengthcol = [1]*len(df1)
df1.insert(loc=0, column='x0', value=lengthcol)
na1 = np.array(df1)
out = np.dot(na1,np.array(wl))
test=100*(len(out[out>0]))/len(out)
print('spam',test,'%')

df1 = pd.DataFrame(columns=nul)
df1 = getDF(df1,testhampath,'ham','LRTest',nul)
lengthcol = [1]*len(df1)
df1.insert(loc=0, column='x0', value=lengthcol)
na1 = np.array(df1)
out = np.dot(na1,np.array(wl))
test=100*(len(out[out<0]))/len(out)
print('ham',test,'%')
t7=dt.datetime.now()
print('time',t7-t6)