import pandas as pd
import numpy as np

def rank_query(data, qid, sort_cols):
    """
    Description: Generates a list of the true labels, ranked by their expected relevance
    for the specified query
    
    Inputs:
        data: Pandas Dataframe with columns {qid, label_true, ERel}, grouped by qid
        qid: query id of query to return
        sort_cols: str or list of columns to sort by, in order of sorting preference 
    
    """
    data = data.get_group(qid)
    return data.sort_values(sort_cols, ascending=False).ix[:,'label_true']