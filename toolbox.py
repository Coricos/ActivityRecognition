# Author : DINDIN Meryll
# Date : 01/11/2017

from imports import *

# Defines the weights corresponding to a given array of labels
def sample_weight(lab) :

    # Defines the sample_weight
    res = np.zeros(len(lab))
    wei = compute_class_weight('balanced', np.unique(lab), lab)
    wei = wei / sum(wei)
    
    for ele in np.unique(lab) :
        for idx in np.where(lab == ele)[0] :
            res[idx] = wei[int(ele)]

    del wei

    return res

# Reformat data according to a problematic
def reformat_vectors(vec, reduced=False, red_index=[6,7]) :

    if not reduced : 
    	return [vec[:,idx,:].reshape(vec.shape[0], vec.shape[2], 1) for idx in range(vec.shape[1])]
    else : 
    	return [vec[:,idx,:].reshape(vec.shape[0], vec.shape[2], 1) for idx in red_index]

# Remove doublon
def remove_doublon(l) :

    new = []
    for ele in l : 
        if ele not in new : new.append(ele)

    return new

# Time efficient clearing in dataframes
def fast_clear(dtf) :

    vec = dtf.values
    vec[vec == -np.inf] = 0
    vec[vec == np.inf] = 0
    vec = np.nan_to_num(vec)

    return pd.DataFrame(vec, columns=dtf.columns, index=dtf.index)

# Time efficient concatenation
def fast_concatenate(list_dtf, axis=1) :

    if len(list_dtf) == 1 : return list_dtf[0]
    else :

        if axis == 1 :
            col = []
            for dtf in list_dtf : col += list(dtf.columns)
            idx = list_dtf[0].index
        elif axis == 0 :
            col = list_dtf[0].columns
            idx = []
            for dtf in list_dtf : idx += list(dtf.index)

        return pd.DataFrame(np.concatenate(tuple([dtf.values for dtf in list_dtf]), axis=axis), columns=np.asarray(col), index=np.asarray(idx))

# Multiprocessed extraction
def extract(couple_index, data) : 

    return data[couple_index[0]:couple_index[1]].values.transpose()

