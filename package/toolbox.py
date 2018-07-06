# Author : DINDIN Meryll
# Date : 01/11/2017

from package.imports import *

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

# Defines a dictionnary composed of class weights
def class_weight(lab) :
    
    res = dict()
    
    for idx, ele in enumerate(compute_class_weight('balanced', np.unique(lab), lab)) : res[idx] = ele
        
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
    idx = [ind for ind in range(len(dtf.columns)) if ind not in tmp]
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
def extract_hapt(couple_index, data) : 

    return data[couple_index[0]:couple_index[1]].values.transpose()

# Multiprocessed extraction
def extract_shl(couple_index, data) :

    tmp = data[couple_index[0]:couple_index[1]].transpose()
    n_a = np.sqrt(np.square(tmp[0]) + np.square(tmp[1]) + np.square(tmp[2]))
    n_g = np.sqrt(np.square(tmp[3]) + np.square(tmp[4]) + np.square(tmp[5]))

    return np.asarray(list(tmp) + [n_a] + [n_g])

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
    return pd.DataFrame(np.asarray(dtf).transpose(), index=['Acc', 'Rec', 'Pre', 'F1S'], columns=['Main'] + ['Class_{}'.format(k) for k in np.unique(lab)])

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

# Multiprocessed computation of the betti curves
def compute_betti_curves(sig):

    from package.topology import Levels

    fil = Levels(sig)

    return fil.betti_curves()

# Easier to call and recreate the channel array
# turn_on refers to the list of channels to turn-on
def generate_channels(turn_on):

    dic = {
           'with_acc_cv2': False,
           'with_acc_cv1': False,
           'with_n_a_cv1': False,
           'with_n_a_tda': False,
           'with_gyr_cv2': False,
           'with_gyr_cv1': False,
           'with_n_g_cv1': False,
           'with_n_g_tda': False,
           'with_qua_cv2': False,
           'with_fea': False,
           'with_fft': False
           }
    
    for key in turn_on: dic[key] = True
    
    return dic

# Multiprocessed way of computing the limits of a persistent diagrams
# vec refers to a 1D numpy array
def persistent_limits(vec):
    
    lvl = Levels(vec)
    u,d = lvl.get_persistence()
    
    return np.asarray([min(u[:,0]), max(u[:,1]), min(d[:,0]), max(d[:,1])])

# Compute the Betti curves
# vec refers to a 1D array
def compute_betti_curves(vec, mnu, mxu, mnd, mxd):

    fil = Levels(vec)
    try: v,w =  lvl.betti_curves(mnu, mxu, mnd, mxd, num_points=100)
    except: v,w = np.zeros(100), np.zeros(100)
    del fil
    
    return np.vstack((v,w))

# Compute the landscapes
# vec refers to a 1D array
def compute_landscapes(vec, mnu, mxu, mnd, mxd):

    fil = Levels(vec)
    try: p,q = fil.landscapes(mnu, mxu, mnd, mxd, num_points=100)
    except: p,q = np.zeros((10,100)), np.zeros((10,100))
    del fil
    
    return np.vstack((p,q))
