import pandas as pd
import numpy as np

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
        barr.append(np.array(arr[::2])[:-1])
#         barr.append(np.array(arr))
    return np.array(barr)
    
def closed_frequent(frequent):
    #Find Closed frequent itemset
    #Dictionay storing itemset with same support count key
    su = frequent.support.unique()#all unique support count
    fredic = {}
    for i in range(len(su)):
        inset = list(frequent.loc[frequent.support ==su[i]]['itemsets'])
        fredic[su[i]] = inset
    #Dictionay storing itemset with  support count <= key
    fredic2 = {}
    for i in range(len(su)):
        inset2 = list(frequent.loc[frequent.support<=su[i]]['itemsets'])
        fredic2[su[i]] = inset2
    
    cl = []
    for index, row in frequent.iterrows():
        isclose = True
        cli = row['itemsets']
        cls = row['support']
        checkset = fredic[cls]
        for i in checkset:
            if (cli!=i):
                if(frozenset.issubset(cli,i)):
                    isclose = False
                    break

        if(isclose):
            cl.append(row['itemsets'])
    return cl


def column_names(X):
    unique_items = set()
    for transaction in X:
        for item in transaction:
            unique_items.add(item)
    columns_ = sorted(unique_items)
    columns_mapping = {}
    for col_idx, item in enumerate(columns_):
        columns_mapping[item] = col_idx
    columns_mapping_ = columns_mapping
    return columns_, columns_mapping

def convert_to_bool_df(X, columns_, columns_mapping_):
    array = np.zeros((len(X), len(columns_)), dtype=bool)
    for row_idx, transaction in enumerate(X):
        for item in transaction:
            col_idx = columns_mapping_[item]
            array[row_idx, col_idx] = True
    return array