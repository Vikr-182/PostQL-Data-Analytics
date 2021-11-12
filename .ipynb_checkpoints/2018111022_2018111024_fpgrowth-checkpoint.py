# mlxtend Machine Learning Library Extensions
# Author: Steve Harenberg <harenbergsd@gmail.com>
#
# License: BSD 3 clause

import math
import itertools

import numpy as np
import pandas as pd
import collections
from collections import defaultdict as Dict
from distutils.version import LooseVersion as Version
from pandas import __version__ as pandas_version


class Node(object):
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = Dict(Node)

        if parent is not None:
            parent.children[item] = self

    def fine_path_from_root(self):
        """ Returns the top-down sequence of items from self to
            (but not including) the root node. """
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path


class ConditionedFPTree(object):
    def __init__(self, ordering=None):
        self.ordering = ordering        
        self.root = Node(None)
        self.nodes = Dict(list)
        self.cond_items = []

    def conditional_tree(self, cond_item, minsup):
        branches = []
        count = Dict(int)
        for node in self.nodes[cond_item]:
            branch = node.fine_path_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count

        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        ordering = {item: i for i, item in enumerate(items)}

        cond_tree = ConditionedFPTree(ordering)
        for idx, branch in enumerate(branches):
            branch = sorted([i for i in branch if i in ordering],
                            key=ordering.get, reverse=True)
            cond_tree.insert_transaction(branch, self.nodes[cond_item][idx].count)
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

    def insert_transaction(self, itemset, count=1):
        self.root.count += count

        if len(itemset) == 0:
            return

        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
                index += 1
            else:
                break

        # Insert any remaining items
        for item in itemset[index:]:
            child_node = Node(item, count, node)
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True

    def print_status(self, count, colnames):
        cond_items = [str(i) for i in self.cond_items]
        if colnames:
            cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ", ".join(cond_items)
        print('\r%d itemset(s) from tree conditioned on items (%s)' %
              (count, cond_items), end="\n")

def setup_fptree(df, min_support):
    num_itemsets = len(df.index)        # number of itemsets in the database
    itemsets = df.values

    # support of each individual item
    # if itemsets is sparse, np.sum returns an np.matrix of shape (1, N)
    item_support = np.array(np.sum(itemsets, axis=0) / float(num_itemsets))
    item_support = item_support.reshape(-1)

    items = np.nonzero(item_support >= min_support)[0]

    # Define ordering on items for inserting into FPTree
    indices = item_support[items].argsort()
    ordering = {item: i for i, item in enumerate(items[indices])}

    # Building tree by inserting itemsets in sorted order
    # Heuristic for reducing tree size is inserting in order
    #   of most frequent to least frequent
    tree = ConditionedFPTree(ordering)
    for i in range(num_itemsets):
        nonnull = np.where(itemsets[i, :])[0]
        itemset = [item for item in nonnull if item in ordering]
        itemset.sort(key=ordering.get, reverse=True)
        tree.insert_transaction(itemset)

    return tree, ordering

def generate_itemsets(generator, num_itemsets, colname_map):
    itemsets = []
    supports = []
    for sup, iset in generator:
        itemsets.append(frozenset(iset))
        supports.append(sup / num_itemsets)

    res_df = pd.DataFrame({'support': supports, 'itemsets': itemsets})

    if colname_map is not None:
        res_df['itemsets'] = res_df['itemsets'] \
            .apply(lambda x: frozenset([colname_map[i] for i in x]))

    return res_df

def valid_input_check(df):
    # Fast path: if all columns are boolean, there is nothing to checks
    all_bools = df.dtypes.apply(pd.api.types.is_bool_dtype).all()
    try:
        if not all_bools:
            values = df.values
            idxs = np.where((values != 1) & (values != 0))
    except:
        print("ERROR IN INPUT")
            
def FPgrowth(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0, merge=False):
    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    tree, _ = setup_fptree(df, min_support)
    minsup = math.ceil(min_support * len(df.index))  # min support as count
    generator = compute_fpg(tree, minsup, colname_map, max_len, verbose,merge)

    return generate_itemsets(generator, len(df.index), colname_map)


def compute_fpg(tree, minsup, colnames, max_len, verbose, merge=False):
    count = 0
    items = tree.nodes.keys()
    if tree.is_path():
        # If the tree is a path, we can combinatorally generate all
        # remaining itemsets without generating additional conditional trees
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                support = min([tree.nodes[i][0].count for i in itemset])
                yield support, tree.cond_items + list(itemset)
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])
            yield support, tree.cond_items + [item]

    if verbose:
        tree.print_status(count, colnames)

    # Generate conditional trees to generate frequent itemsets one item larger and merge based on path:wq
    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)
            for sup, iset in compute_fpg(cond_tree, minsup,
                                      colnames, max_len, verbose):
                yield sup, iset

                
import pandas as pd
import numpy as np
from utils import *
from mlxtend.frequent_patterns import fpgrowth, apriori # fpgrowth, apriori library                
dataset = parse_data("data/Skin.txt")
columns_, columns_mapping = column_names(dataset)
array = convert_to_bool_df(dataset, columns_, columns_mapping)
df = pd.DataFrame(array, columns=columns_)                
                
import time
start_time = time.time()
frequent_itemsets = FPgrowth(df, min_support=0.2)
closed_frequent_itemsets = closed_frequent(frequent_itemsets)
print(closed_frequent_itemsets)
print('Time to find Close frequent itemset using Library')
print("--- %s seconds ---" % (time.time() - start_time))
