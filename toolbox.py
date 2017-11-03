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

# Read labels and users for features
def read_text_file(path, column) :

    with open('{}'.format(path), 'r') as raw : res = raw.readlines()
    for ind in range(len(res)) : res[ind] = res[ind].replace('\n', '')

    return pd.DataFrame(np.asarray(res).astype(int), columns=[column])

# Defines the number of possible sliced windows
def windows(size, time_window, overlap) :

    cnt, srt = 0, 0
    while srt <= size - time_window :
        cnt += 1
        srt += int(overlap * time_window)

    return cnt  

# Reformat data according to a problematic
def reformat_vectors(vec, mod, reduced=False, red_index=[6,7]) :

    if mod in ['Conv1D', 'DeepConv1D'] :
        if not reduced : return [vec[:,idx,:].reshape(vec.shape[0], vec.shape[2], 1) for idx in range(vec.shape[1])]
        else : return [vec[:,idx,:].reshape(vec.shape[0], vec.shape[2], 1) for idx in red_index]
    
    elif mod in ['Conv2D', 'DeepConv2D'] :
        return vec.reshape(vec.shape[0], 1, vec.shape[1], vec.shape[2])

# Remove doublon
def remove_doublon(l) :

    new = []
    for ele in l : 
        if ele not in new : new.append(ele)

    return new

# Time efficient function for columns extraction
def extract_columns(dtf, col) :
    
    # Index to extract
    idx = [ind for ind, ele in enumerate(dtf.columns) if ele in col]
    # Extract DataFrame
    val = dtf.values[:, idx]
    dtf = pd.DataFrame(val, index=dtf.index, columns=col)
    # Memory efficiency
    del idx, val
    
    return dtf  

# Time efficient function for columns deletion
def remove_columns(dtf, col) :
    
    # Defines the indexes to remove
    tmp = [ind for ind, ele in enumerate(dtf.columns) if ele in col]
    idx = [ind for ind in xrange(len(dtf.columns)) if ind not in tmp]
    # Extract DataFrame
    val = dtf.values[:, idx]
    dtf = pd.DataFrame(val, index=dtf.index, columns=np.asarray(dtf.columns)[idx])
    # Memory efficiency
    del idx, val
    
    return dtf 

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

# Function aiming at displaying scores
def score_verbose(y_true, y_pred) :

    dtf = []
    # Compute the mean scores
    acc = accuracy_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    pre = precision_score(y_true, y_pred, average='weighted')
    dtf.append([acc, rec, pre, f1s])
    # Relative results to each class
    lab = np.unique(list(np.unique(y_true)) + list(np.unique(y_pred)))
    y_t = preprocessing.label_binarize(y_true, np.unique(lab), pos_label=1)
    y_p = preprocessing.label_binarize(y_pred, np.unique(lab), pos_label=1)
    for ind in range(len(lab)) :
        # Common binary costs
        pre = precision_score(y_t[:,ind], y_p[:,ind], sample_weight=sample_weight(y_t[:,ind]))
        rec = recall_score(y_t[:,ind], y_p[:,ind], sample_weight=sample_weight(y_t[:,ind]))
        f1s = f1_score(y_t[:,ind], y_p[:,ind], sample_weight=sample_weight(y_t[:,ind]))
        acc = accuracy_score(y_t[:,ind], y_p[:,ind], sample_weight=sample_weight(y_t[:,ind]))
        dtf.append([acc, rec, pre, f1s])
    # Memory efficiency
    del acc, f1s, rec, pre, lab, y_t, y_p
    # Return dataframe for score per class
    return pd.DataFrame(np.asarray(dtf).transpose(), index=['Acc', 'Rec', 'Pre', 'F1S'], columns=['Main'] + ['Class_{}'.format(k) for k in range(len(np.unique(y_pred)))])
