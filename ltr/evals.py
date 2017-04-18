import numpy as np
import pandas as pd
import math
import ltr.logistic_classifier


def dcg_at_rank_N(r, rank=10, non_linear_gain='y'):
    """
    Description: Discounted Cumulative Gain (DCG) computation
        
    Inputs:
    r - Ordered relevance scores (list)
    rank - Top [rank] number of results to use
    non_linear_gain - flag (y/n) for whether to use a non_linear gain function

    """
    
    r = np.asfarray(r)
    top_N = r[:rank]
    if non_linear_gain=='n':
        discount = 1/np.log2(np.arange(2, len(top_N)+1))
        dcg =  top_N[0] + np.sum(top_N[1:] * discount)
    elif non_linear_gain=='y':
        gain = 2**(top_N) - 1
        discount = 1/np.log2(np.arange(2, len(top_N)+2))
        dcg =  np.sum(gain * discount) 
    return dcg

def ndcg_at_rank_N(r, rank=10, non_linear_gain='y'):
    """
    Description: Normalised Discounted Cumulative Gain (NDCG) computation -
        DCG score normalised by the best possible DCG score for the relevance
        scores
    
    Inputs:
    r - Ordered relevance scores (list)
    rank - Top [rank] number of results to use
    non_linear_gain - flag (y/n) for whether to use a non_linear gain function

    """
    
    best_dcg = dcg_at_rank_N(sorted(r, reverse=True), rank, non_linear_gain)
    if best_dcg:
        dcg = dcg_at_rank_N(r, rank, non_linear_gain)
        normalised_dcg = dcg/best_dcg
        return normalised_dcg
    else:
        return 0

def mean_ndcg(data, rank=10, non_linear_gain='y'):
    """
    Description: Average (mean) of Normalized Discounted Cumulative Gain (NDGC)
        across all queries in data
    
    Inputs:
    data - Pandas Dataframe with columns {qid, label_true, ERel}, grouped by qid
    rank - Top [rank] number of results to use
    non_linear_gain - flag (y/n) for whether to use a non_linear gain function
    
    """
    ndcg_list = []
    for qid,_ in data:
        ndcg_list.append(ndcg_at_rank_N(ltr.logistic_classifier.rank_query(data=data, qid=qid), rank, non_linear_gain))
    return np.mean(ndcg_list)

def err(r, rank=10):
    """
    Description: Expected Reciprocal Rank (ERR) at rank computation,
        computes in linear time
    
    Inputs:
    r - Ordered relevance scores (list)
    rank - Top [rank] number of results to use
    
    """
    ERR = 0.0
    p = 1.0
    if rank==None:
        rank = len(r)
    else:
        rank = np.min([rank,len(r)])
        
    for x in range(0,rank):
        g = r.iloc[x] + 1
        Rg = float(2**g - 1)/float(2**5)
        ERR = ERR + p*Rg/(x+1)
        p = p*(1-Rg)
        
    return ERR

def mean_err(data):
    """
    Description: Average (mean) of Expected Reciprocal Rank
    
    Inputs:
    data - Pandas Dataframe with columns {qid, label_true, ERel}, grouped by qid
    
    """
    err_list = []
    for qid,_ in data:
        err_list.append(err(ltr.logistic_classifier.rank_query(data=data, qid=qid), rank=10))
    return np.mean(err_list)
