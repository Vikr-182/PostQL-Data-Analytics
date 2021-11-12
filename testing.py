import pandas as pd
import numpy as np
from utils import *
from FPgrowth import FPgrowth # self implemented fpgrowth
from mlxtend.frequent_patterns import fpgrowth, apriori # fpgrowth, apriori library

dataset = parse_data("data/Skin.txt")
columns_, columns_mapping = column_names(dataset)
array = convert_to_bool_df(dataset, columns_, columns_mapping)
df = pd.DataFrame(array, columns=columns_)

'''
# Task 1: Apriori Library
import time
start_time = time.time()
frequent_itemsets = apriori(df, min_support=0.2)
closed_frequent_itemsets = closed_frequent(frequent_itemsets)
print('Time to find Close frequent itemset in Apriori Library')
print("--- %s seconds ---" % (time.time() - start_time))

# Task 2: Apriori Baseline without optimization
import time
start_time = time.time()
frequent_itemsets = FPgrowth(df, min_support=0.2)
closed_frequent_itemsets = closed_frequent(frequent_itemsets)
print('Time to find Close frequent itemset in Baseline without optimization')
print("--- %s seconds ---" % (time.time() - start_time))

# Task 3: Apriori with Hash based
import time
start_time = time.time()
frequent_itemsets = fpgrowth(df, min_support=0.2)
closed_frequent_itemsets = closed_frequent(frequent_itemsets)
print('Time to find Close frequent itemset using Library')
print("--- %s seconds ---" % (time.time() - start_time))


# Task 4: Apriori with Partitioning
import time
start_time = time.time()
frequent_itemsets = partition(records,3,0.2)
closed_frequent_itemsets = closed_frequent(frequent_itemsets)
print('Time to find Close frequent itemset using Library')
print("--- %s seconds ---" % (time.time() - start_time))


# Task 5: FP Growth Library
import time
start_time = time.time()
frequent_itemsets = fpgrowth(df, min_support=0.2)
closed_frequent_itemsets = closed_frequent(frequent_itemsets)
print('Time to find Close frequent itemset using Library')
print("--- %s seconds ---" % (time.time() - start_time))

# Task 6: FP Growth Baseline without optimization
import time
start_time = time.time()
frequent_itemsets = FPgrowth(df, min_support=0.2)
closed_frequent_itemsets = closed_frequent(frequent_itemsets)
print('Time to find Close frequent itemset baseline')
print("--- %s seconds ---" % (time.time() - start_time))

# Task 7: FP Growth with Merging
import time
start_time = time.time()
frequent_itemsets = fpgrowth(df, min_support=0.2, merge=True)
closed_frequent_itemsets = closed_frequent(frequent_itemsets)
print('Time to find Close frequent itemset baseline with merging')
print("--- %s seconds ---" % (time.time() - start_time))

'''
