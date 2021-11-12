
import itertools
from collections import defaultdict

thresh = 2


def def_value():
    return 0


def iter1(records):
    candidates = defaultdict(def_value)
    final = defaultdict(def_value)
    for i in records:
        for element in i :
            candidates[element] += 1

    for key,value in candidates.items():
        if value >= thresh:
            final[key] = value

    return final



def iter2(final1,records):
    candidates = {}
    final = defaultdict(def_value)
    for i in itertools.combinations(final1,2):
        candidates[i] = 0

        for j in records :
            if set(i).issubset(set(j)):
                candidates[i]+=1
    
    for key,value in candidates.items():
        if value >= thresh:
            final[key] = value

    return final


def iterhash(records):
    candidates = defaultdict(def_value)
    candidates2 = defaultdict(def_value)
    final = defaultdict(def_value)
    final2 = defaultdict(def_value)
    for i in records:
        for element in i :
            candidates[element] += 1
        for element in itertools.combinations(i,2):
            candidates2[element] += 1

    for key,value in candidates.items():
        if value >= thresh:
            final[key] = value


    for key,value in candidates2.items():
        if value >= thresh:
            final2[key] = value

    return final,final2

def pruning(currset,prevset,lev):
    newdict = {}
    
    #print ("yaar ", set(prevset.keys()))
    for i in currset:
        iflag = True
        for subset in itertools.combinations(i,lev-1):
            if  subset not in list(prevset.keys()):
                iflag = False 
                #print (set(subset),"::::",lev,"::::",currset)
                #print (set(prevset.keys()))
                #print ("hua hi nahi yar")


        if iflag == True :
            newdict[i] = 0 

    return newdict 

def itern(n,finalprev,records):
    
    final = {}
    allnum = sorted(list(set([k for p in finalprev.keys() for k in p])))
    allc = list(itertools.combinations(allnum,n))
    
    print ("before pruning")
    print (allc)

    candidates = pruning (list(allc),finalprev,n)

    print ("after pruning")
    print (candidates)

    for i in candidates:
        for j in records :

            if set(i).issubset(set(j)):
                candidates[i] += 1

    for key,value in candidates.items():
        if value >= thresh:
            final[key] = value

    return final



records = [[1,2,5],[2,4],[2,3],[1,2,4],[1,3],[2,3],[1,3],[1,2,3,5],[1,2,3]]


emp  = {}

lent = 3

d1 = iter1(records)


emp = {**emp,**d1}

print ("of length 1")
print (d1)

d2 = iter2(d1,records)


a1,a2 = iterhash(records)

print (a1)


print ("of length 2")
print (d2)

print (a2)

last = d2



emp = {**emp,**d2}

while len(last) != 0:
    upd = itern(lent,last,records) 
    print ("of length ",lent)
    print (upd)
    emp = {**emp,**upd}
    last = upd
    lent += 1



print (emp)



