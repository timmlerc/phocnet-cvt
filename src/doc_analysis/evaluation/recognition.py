'''
Created on Jan 9, 2019

@author: fwolf 
'''

import numpy as np
from scipy.spatial.distance import cdist
#from doc_analysis.dap import _dap

def er_from_query_lexicon_feature_matrices(query_features,
                                           lexicon_features,
                                           qry_labels,
                                           lexicon_labels,
                                           metric = 'cosine'):
    
    if metric == 'cosine':
        dist_mat = cdist(XA=query_features, XB=lexicon_features, metric='cosine')
        dist_min = np.argmin(dist_mat, axis=1)

        class_ids = lexicon_labels[dist_min]
    else:
        raise ValueError('Unknown metric for wer from lexicon features')

    relevance_vec = (class_ids == qry_labels)
    wer = 1 - (np.sum(relevance_vec)/float(len(relevance_vec))) 

    return wer, class_ids

