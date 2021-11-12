
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd



def def_value():
    return 0


def parse_data(file_name):

    file = open(file_name)
    barr = []
    for ind, line in enumerate(file):
        arr = []
        for i in line.split(" "):
            try:
                arr.append(int(i))
            except:
                pass
        #barr.append(list(np.array(arr[::2])[:-1]))
        barr.append(arr)

    return barr


def iter1(records,thresh):
    candidates = defaultdict(def_value)
    final = defaultdict(def_value)
    for i in records:
 
        i = list(set(i))

        for element in i :
            candidates[element] += 1

    for key,value in candidates.items():
        if value >= thresh:
            final[(key,)] = value

    return final







def iter2(final1,records,thresh):
    candidates = {}
    final = defaultdict(def_value)
    listf = [x[0] for x in final1]
    for i in itertools.combinations(listf,2):
        candidates[i] = 0

        for j in records :
            if set(i).issubset(set(j)):
                candidates[i]+=1
    
    for key,value in candidates.items():
        if value >= thresh:
            final[key] = value

    return final


def iterhash(records,thresh):
    candidates = defaultdict(def_value)
    candidates2 = defaultdict(def_value)
    final = defaultdict(def_value)
    final2 = defaultdict(def_value)
    for i in records:

        i = list(set(i))


        for element in i :
            candidates[element] += 1
        for element in itertools.combinations(i,2):
            candidates2[element] += 1

    for key,value in candidates.items():
        if value >= thresh:
            final[(key,)] = value


    for key,value in candidates2.items():
        if value >= thresh:
            final2[key] = value

    return final,final2

def pruning(currset,prevset,lev):
    newdict = {}
    for i in currset:
        iflag = True
        for subset in itertools.combinations(i,lev-1):
            if  subset not in list(prevset.keys()):
                iflag = False 

        if iflag == True :
            newdict[i] = 0 

    return newdict 

def itern(n,finalprev,records,thresh):
    
    final = {}
    allnum = sorted(list(set([k for p in finalprev.keys() for k in p])))
    allc = list(itertools.combinations(allnum,n))
    
    ###print ("before pruning")
    #print (allc)

    candidates = pruning (list(allc),finalprev,n)

    #print ("after pruning")
    #print (candidates)

    for i in candidates:
        for j in records :

            if set(i).issubset(set(j)):
                candidates[i] += 1

    for key,value in candidates.items():
        if value >= thresh:
            final[key] = value

    return final

from utils import *

def parse_data(file_name):

    file = open(file_name)
    barr = []
    for ind, line in enumerate(file):
        arr = []
        for i in line.split(" "):
            try:
                arr.append(int(i))
            except:
                pass
        barr.append(list(np.array(arr[::2])[:-1]))
    return barr

<<<<<<< HEAD

records = parse_data("data/Skin.txt")
# records = [[1,2,5],[2,4],[2,3],[1,2,4],[1,3],[2,3],[1,3],[1,2,3,5],[1,2,3]]
=======
print ("processing data .." )
records = parse_data("data/Skin.txt")





def dic2pd(retdict,tot):
    newlist = []
    for i in retdict:
        newlist.append({'itemset':i,'support':retdict[i]/tot})
    return pd.DataFrame.from_dict(newlist)
>>>>>>> 73afca44aebefb85c20b63beb2afb270605f3573





def apriori(records,sup):
    emp  = {}
    lent = 3
    thresh = sup*len(records)
    d1 = iter1(records,thresh)


    emp = {**emp,**d1}

    print ("of length 1")
    print (d1)

    d2 = iter2(d1,records,thresh)

    #a1,a2 = iterhash(records,thresh)
    #print (a1)

    print ("of length 2")
    print (d2)
    #print (a2)
    last = d2

    emp = {**emp,**d2}

    while len(last) != 0:
        upd = itern(lent,last,records,thresh) 
        print ("of length ",lent)
        print (upd)
        emp = {**emp,**upd}
        last = upd
        lent += 1



    print (emp)
    return dic2pd(emp,len(records))








def apriori_hash(records,sup):
    emp  = {}
    lent = 3
    thresh = sup*len(records)

    d1,d2 = iterhash(records,thresh)
    
    print ("of length 1")
    print (d1)



    print ("of length 2")
    print (d2)
    last = d2

    emp = {**emp,**d1}
    emp = {**emp,**d2}

    while len(last) != 0:
        upd = itern(lent,last,records,thresh) 
        print ("of length ",lent)
        print (upd)
        emp = {**emp,**upd}
        last = upd
        lent += 1



<<<<<<< HEAD
emp = {**emp,**d2}
=======
    #print (emp)
    return dic2pd(emp,len(records))

#st_hash = 
print (apriori_hash(records,0.2))
print (apriori(records,0.2))

>>>>>>> 73afca44aebefb85c20b63beb2afb270605f3573


<<<<<<< HEAD
print (emp)
=======





>>>>>>> 73afca44aebefb85c20b63beb2afb270605f3573
