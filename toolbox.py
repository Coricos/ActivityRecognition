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

# Reformat the vectors : dynamic model
def reformat(vec, typ) :

    if typ == '1D' : return vec.reshape(vec.shape[0], vec.shape[1], 1)
    elif typ == '2D' : return vec.reshape(vec.shape[0], 1, vec.shape[1], vec.shape[2])

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

# Clearing arrays
def clear_array(arr) :

    arr[arr == -np.inf] = 0
    arr[arr == np.inf] = 0
    arr = np.nan_to_num(arr)

    return arr 

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
    del acc, f1s, rec, pre, y_t, y_p
    # Return dataframe for score per class
    return pd.DataFrame(np.asarray(dtf).transpose(), index=['Acc', 'Rec', 'Pre', 'F1S'], columns=['Main'] + ['Class_{}'.format(k) for k in range(len(np.unique(lab)))])

# From vector to movement, on multiprocessed way
def from_vec_to_mvt(vec, sampling_frequency=50) :
    
    col = ['Time', 'Battery', 'Temp', 'Altitude', 'Ga1', 'Ga2', 'Ga3', 'Om1', 'Om2', 'Om3', 'Ma1', 'Ma2', 'Ma3']
    tmp = np.random.rand(vec.shape[1]).reshape(1, vec.shape[1])
    vec = np.vstack([tmp, tmp, tmp, tmp, vec[:6], tmp, tmp, tmp])
    dtf = pd.DataFrame(vec.transpose(), columns=col)
    sam = Sample('None', dtf=dtf)
    sam.sampling_frequency = 50
    
    return Movement(sam)

# Defines a way to truncate the given problematic
def truncate_data(vec, lab, lab_to_del=[6, 7, 8, 9, 10, 11]) :

    # Defines the mask
    msk = np.ones(len(lab), dtype=bool)
    # Incremental deletion
    for val in lab_to_del : msk[np.where(lab == val)[0]] = False

    return [ele[msk] for ele in vec], lab[msk]

# Get the mask to spread it, same as above
def get_mask(lab, lab_to_del=[6, 7, 8, 9, 10, 11]) :
    
    # Defines the mask
    msk = np.ones(len(lab), dtype=bool)
    # Incremental deletion
    for val in lab_to_del : msk[np.where(lab == val)[0]] = False
        
    return msk

# Multiprocessed fft computation
def multi_fft(vec) :

    return np.abs(np.fft.rfft(vec, axis=0))

# Multiprocessed computation of quaternions
def compute_quaternion(sig):

    dT = 1.0/50.0
    quaternion = np.zeros((4, sig.shape[1]))
    quaternion[0,0] = 1

    for j in range(1, sig.shape[1]) :
        r = quaternion[:,j-1]
        q = np.array([1, dT*sig[0,j]*math.pi/360., dT*sig[1,j]*math.pi/360., dT*sig[2,j]*math.pi/360.])
        quaternion[0,j] = (r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3])
        quaternion[1,j] = (r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2])
        quaternion[2,j] = (r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1])
        quaternion[3,j] = (r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0])
                
    return quaternion
