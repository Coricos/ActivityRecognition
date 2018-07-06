# Author : DINDIN Meryll
# Date : 01/11/2017

from package.toolbox import *

# Build a specific loader for the HAPT Dataset matching features with their raw signals
class HAPT_Loader :

    # Initialization
    # output refers to where to create the new dataset
    # storage refers to where to fetch the input data
    # max_jobs reflects the amount of threads to launch concurrently
    def __init__(self, output='./dataset/HAPT_ini.h5', storage='./HAPT', max_jobs=multiprocessing.cpu_count()) :

        # Cares about multiprocessing instances
        self.njobs = max_jobs
        # Match the conditions of the initial paper
        self.usr_train = np.unique(read_text_file('{}/subject_id_train.txt'.format(storage), 'Subjects').values)
        self.usr_valid = np.unique(read_text_file('{}/subject_id_test.txt'.format(storage), 'Subjects').values)
        self.usr_train = [ele for ele in range(1, 31) if ele not in self.usr_valid]
        self.time_window = 128
        self.overlap_rto = 0.5
        # Path for serialization
        self.storage = storage
        self.path = output

    # Load the features relative to the signals
    def load_fea(self) :

        # Load the features names
        with open('{}/features.txt'.format(self.storage)) as raw : 
            lab = raw.readlines()
        for ind in range(len(lab)) : 
            tmp = str(lab[ind].replace('\n','').replace(' ',''))
            if tmp in lab : tmp = tmp.replace('1', '2')
            if tmp in lab : tmp = tmp.replace('2', '3')
            lab[ind] = tmp
        
        # Training set
        X_tr = pd.read_csv('{}/X_train.txt'.format(self.storage), sep='\n', delimiter=' ', 
                           header=None, keep_default_na=False, dtype=np.float32)
        X_tr.columns = lab
        l_tr = read_text_file('{}/y_train.txt'.format(self.storage), 'Labels')
        i_tr = read_text_file('{}/subject_id_train.txt'.format(self.storage), 'Subjects')
        # Validation set
        X_va = pd.read_csv('{}/X_test.txt'.format(self.storage), sep='\n', delimiter=' ', 
                           header=None, keep_default_na=False, dtype=np.float32)
        X_va.columns = lab
        l_va = read_text_file('{}/y_test.txt'.format(self.storage), 'Labels')
        i_va = read_text_file('{}/subject_id_test.txt'.format(self.storage), 'Subjects')

        # Save as attribute
        self.train = fast_concatenate([X_tr, l_tr, i_tr], axis=1)
        self.valid = fast_concatenate([X_va, l_va, i_va], axis=1)
        # Memory efficiency
        del X_va, l_va, i_va, lab, X_tr, l_tr, i_tr, raw

        # Serialize the features in database
        with h5py.File(self.path, 'w') as dtb :
            dtb.create_dataset('fea_t', data=remove_columns(self.train, ['Subjects', 'Labels']))
            dtb.create_dataset('fea_v', data=remove_columns(self.valid, ['Subjects', 'Labels']))
        # Memory efficiency
        del self.train, self.valid

    # Loads the raw signals as dataframe
    def load_signals(self) :

        # Where to gather the constructed dataframes
        raw = []
        # Extracts iteratively
        for fle in tqdm.tqdm(['_'.join(ele.split('_')[1:]) for ele in sorted(glob.glob('{}/acc_*'.format(self.storage)))]):
            try : 
                # Load the accelerometer data
                acc = pd.read_csv('{}/acc_{}'.format(self.storage, fle), sep='\n', delimiter=' ', 
                                  header=None, keep_default_na=False, dtype=np.float32)
                acc.columns = ['Acc_x', 'Acc_y', 'Acc_z']
                # Load the gyrometer data
                gyr = pd.read_csv('{}/gyro_{}'.format(self.storage, fle), sep='\n', delimiter=' ', 
                                  header=None, keep_default_na=False, dtype=np.float32)
                gyr.columns = ['Gyr_x', 'Gyr_y', 'Gyr_z']
                # Load the metadata
                exp = pd.DataFrame(np.asarray([int(fle.split('exp')[1][:2]) for i in range(len(acc))]), columns=['Experience'])
                usr = pd.DataFrame(np.asarray([int(fle.split('user')[1][:2]) for i in range(len(acc))]), columns=['User'])
                # Build the dataframe
                raw.append(fast_concatenate([exp, usr, acc, gyr], axis=1))
                # Memory efficiency
                del acc, gyr, exp, usr
            except : pass

        # Concatenate every obtained dataframe
        raw = fast_concatenate(raw, axis=0)
        # Build the norms (referential independance)
        raw['Normed_A'] = np.sqrt(np.square(raw['Acc_x'].values) + np.square(raw['Acc_y']) + np.square(raw['Acc_z']))
        raw['Normed_G'] = np.sqrt(np.square(raw['Gyr_x'].values) + np.square(raw['Gyr_y']) + np.square(raw['Gyr_z']))
        # Build the labels
        lab = pd.read_csv('{}/labels.txt'.format(self.storage), sep='\n', delimiter=' ', header=None)
        lab.columns = ['Experience', 'User', 'Label', 'Begin', 'End']
        # Save as attributes
        self.raw_signals = raw
        self.description = lab
        # Memory efficiency
        del raw, lab

    # Slice the signal accordingly to the time_window and overlap
    def load_raw(self) :

        # First, load the signals
        if not hasattr(self, 'raw_signals'): self.load_signals()

        # Local function for slicing
        def slice_signal(sig) :

            if len(sig) < self.time_window: return []
            else:
                # Init variables
                tme, srt, top = [], 0, len(sig)
                # Prepares multiprocessing
                while srt <= top - self.time_window:
                    tme.append((srt, srt + self.time_window))
                    srt += int(self.overlap_rto * self.time_window)
                # Launch multiprocessing
                pol = multiprocessing.Pool(processes=min(len(tme), self.njobs))
                mvs = pol.map(partial(extract_hapt, data=sig), tme)
                pol.close()
                pol.join()
                # Memory efficiency
                del tme, srt, top, pol
                # Return the sliced signals
                return mvs

        # Where to gather the new arrays
        X_tr, y_tr, X_va, y_va = [], [], [], []

        # Deals with the training set
        for ids in tqdm.tqdm(self.usr_train):
            for exp in np.unique(self.description.query('User == {}'.format(ids))['Experience']) :
                cut = self.description.query('Experience == {} & User == {}'.format(exp, ids))
                for val in cut[['Label', 'Begin', 'End']].values :
                    tmp = self.raw_signals.query('Experience == {} & User == {}'.format(exp, ids))
                    sig = slice_signal(remove_columns(tmp[val[1]:val[2]+1], ['Experience', 'User']))
                    y_tr += list(np.full(len(sig), val[0]))
                    X_tr += sig
                    del tmp, sig
                del cut

        # Deals with the validation set
        for ids in tqdm.tqdm(self.usr_valid):
            for exp in np.unique(self.description.query('User == {}'.format(ids))['Experience']) :
                cut = self.description.query('Experience == {} & User == {}'.format(exp, ids))
                for val in cut[['Label', 'Begin', 'End']].values :
                    tmp = self.raw_signals.query('Experience == {} & User == {}'.format(exp, ids))
                    sig = slice_signal(remove_columns(tmp[val[1]:val[2]+1], ['Experience', 'User']))
                    y_va += list(np.full(len(sig), val[0]))
                    X_va += sig
                    del tmp, sig
                del cut

        # Serialize the features in database
        with h5py.File(self.path, 'r+') as dtb:
            dtb.create_dataset('acc_x_t', data=np.asarray(X_tr)[:,0,:])
            dtb.create_dataset('acc_x_v', data=np.asarray(X_va)[:,0,:])
            dtb.create_dataset('acc_y_t', data=np.asarray(X_tr)[:,1,:])
            dtb.create_dataset('acc_y_v', data=np.asarray(X_va)[:,1,:])
            dtb.create_dataset('acc_z_t', data=np.asarray(X_tr)[:,2,:])
            dtb.create_dataset('acc_z_v', data=np.asarray(X_va)[:,2,:])
            dtb.create_dataset('gyr_x_t', data=np.asarray(X_tr)[:,3,:])
            dtb.create_dataset('gyr_x_v', data=np.asarray(X_va)[:,3,:])
            dtb.create_dataset('gyr_y_t', data=np.asarray(X_tr)[:,4,:])
            dtb.create_dataset('gyr_y_v', data=np.asarray(X_va)[:,4,:])
            dtb.create_dataset('gyr_z_t', data=np.asarray(X_tr)[:,5,:])
            dtb.create_dataset('gyr_z_v', data=np.asarray(X_va)[:,5,:])
            dtb.create_dataset('n_acc_t', data=np.asarray(X_tr)[:,6,:])
            dtb.create_dataset('n_acc_v', data=np.asarray(X_va)[:,6,:])
            dtb.create_dataset('n_gyr_t', data=np.asarray(X_tr)[:,7,:])
            dtb.create_dataset('n_gyr_v', data=np.asarray(X_va)[:,7,:])
            dtb.create_dataset('label_t', data=np.asarray(y_tr).astype(int) - 1)
            dtb.create_dataset('label_v', data=np.asarray(y_va).astype(int) - 1)
        # Memory efficiency
        del X_tr, X_va, y_tr, y_va, self.raw_signals, self.description

# Build a specific loader for the SHL Dataset
class SHL_Loader:

    # Initialization
    def __init__(self, path, time_window, overlap, anatomy):

        # Serialization path
        self.path = path
        # Attributes relative to slicing
        self.time_window = time_window
        self.overlap = overlap
        self.anatomy = anatomy
        self.njobs = multiprocessing.cpu_count()

    # Load the raw signals as dataframe
    def load_signals(self):

        # Local function for slicing
        def slice_signal(sig) :

            if len(sig) < self.time_window : return []
            else :
                # Init variables
                tme, srt, top = [], 0, len(sig)
                # Prepares multiprocessing
                while srt <= top - self.time_window :
                    tme.append((srt, srt + self.time_window))
                    srt += int((1 - self.overlap) * self.time_window)
                # Launch multiprocessing
                pol = multiprocessing.Pool(processes=min(len(tme), self.njobs))
                mvs = pol.map(partial(extract_shl, data=sig), tme)
                pol.close()
                pol.join()
                # Memory efficiency
                del tme, srt, top, pol
                # Return the sliced signals
                return mvs

        # Defines the main directory gathering the data
        root_path = '/home/ubuntu/HackATon/Data/TrainingData/SHLDataset_preview_v1/'
        # Launch the scrapping
        for usr in ['User1', 'User2', 'User3'] :
            # Where to temporary gather the data
            mvs, lbl = [], []
            # Loop over dates
            for dry in [ele for ele in os.listdir(root_path + usr) if os.path.isdir('{}/{}'.format(root_path + usr, ele))] :
                pth = root_path + '{}/{}/'.format(usr, dry)
                # Retrieve the values corresponding to the anatomy
                dtf = pd.read_csv(pth + '{}_Motion.txt'.format(self.anatomy), sep='\n', delimiter=' ', header=None, keep_default_na=True)
                dtf = dtf[[0,1,2,3,4,5,6]]
                dtf.fillna(method='pad', limit=3)
                dtf[0] = np.round(dtf[0].values).astype('int64')
                dtf = dtf.values[:,:7].astype('float')
                dtf = np.nan_to_num(dtf)
                # Load the corresponding labels                
                lab = pd.read_csv(pth + 'Label.txt', sep='\n', delimiter=' ', header=None, keep_default_na=True)
                lab = lab.values[:,:2]
                # Slice the signals according to the labels
                idx = np.split(range(lab.shape[0]), np.where(np.diff(lab[:,1]) != 0)[0] + 1)
                for ind, ele in enumerate(idx) :
                    # Delete the weird phases in between activities
                    if np.unique(lab[:,1][ele])[0] == 0 : pass
                    else : 
                        tmp = slice_signal(dtf[ele,1:7])
                        mvs += tmp
                        lbl += list(np.full(len(tmp), np.unique(lab[:,1][ele])[0]))
                        del tmp
                # Memory efficiency
                del idx, lab, dtf, pth
            # Change format
            mvs, lbl = np.asarray(mvs), np.asarray(lbl)
            # Serialize the results for the given user
            with h5py.File(self.path, 'a') as dtb :
                dtb.create_group(usr)
                dtb[usr].create_dataset('ACC', data=np.asarray(mvs)[:,0:3,:])
                dtb[usr].create_dataset('GYR', data=np.asarray(mvs)[:,3:6,:])
                dtb[usr].create_dataset('N_A', data=np.asarray(mvs)[:,6,:])
                dtb[usr].create_dataset('N_G', data=np.asarray(mvs)[:,7,:])
                dtb[usr].create_dataset('y', data=np.asarray(lbl).astype(int))

    # Defines the testing set
    def define_test(self, events_per_label=1):

        # Refers to the created database
        with h5py.File(self.path, 'a') as dtb :
            # Extract an amount of events for each user
            for usr in dtb.keys() :
                # Get the labels
                lab = dtb[usr]['y'].value
                # Split the labels
                idx = np.split(range(lab.shape[0]), np.where(np.diff(lab) != 0)[0] + 1)
                msk = np.ones(lab.shape[0], dtype=bool)
                lbl = np.asarray([np.unique(lab[ele]) for ele in idx])
                # Extraction through mask
                for val in np.unique(lbl) :
                    for ele in np.random.choice(np.where(lbl == val)[0], size=min(events_per_label, len(np.where(lbl == val)[0])), replace=False) :
                        msk[idx[ele]] = False
                # Memory efficiency
                del lab, idx, lbl
                # Serializing new results
                for key in ['ACC', 'GYR', 'N_A', 'N_G', 'y'] :
                    tmp = dtb[usr][key].value
                    dtb[usr].create_dataset(key + '_t', data=tmp[msk])
                    dtb[usr].create_dataset(key + '_e', data=tmp[np.invert(msk)])
                    del dtb[usr][key]
        # Gathers the results
        with h5py.File(self.path, 'a') as dtb :
            # Extract an amount of events for each user
            for key in ['ACC', 'GYR', 'N_A', 'N_G'] :
                dtb.create_dataset(key + '_t', data=np.vstack([dtb[usr][key + '_t'] for usr in ['User1', 'User2', 'User3']]))
                dtb.create_dataset(key + '_e', data=np.vstack([dtb[usr][key + '_e'] for usr in ['User1', 'User2', 'User3']]))
            # Handling labels
            dtb.create_dataset('y_train', data=np.concatenate([dtb[usr]['y_t'] for usr in ['User1', 'User2', 'User3']]))
            dtb.create_dataset('y_valid', data=np.concatenate([dtb[usr]['y_e'] for usr in ['User1', 'User2', 'User3']]))
            # Memory efficiency
            for usr in ['User1', 'User2', 'User3'] : del dtb[usr]

# Build a way to add features vectors
class Constructor:

    # Initialization
    # inp_dtb refers to the input database
    # max_jobs refers to the maximum amount of concurrent threads
    def __init__(self, inp_dtb, max_jobs=multiprocessing.cpu_count()):

        # Defines on which database the work will be done
        self.path = inp_dtb
        # Defines the pool size for multiprocessing
        self.njobs = max_jobs

    # Computes the fft of each normed inertial signal
    def build_fft(self):

        for typ in ['acc', 'gyr']:
            for knd in ['t', 'v']:

                inp = 'n_{}_{}'.format(typ, knd)
                out = 'fft_{}_{}'.format(typ[0], knd)
                with h5py.File(self.path, 'r') as dtb: val = dtb[inp].value
                # Multiprocessed computation
                pol = multiprocessing.Pool(processes=self.njobs)
                fft = np.asarray(pol.map(multi_fft, val))
                pol.close()
                pol.join()
                # Serialize the output
                with h5py.File(self.path, 'a') as dtb :
                    if dtb.get(out): del dtb[out]
                    dtb.create_dataset(out, data=fft)
                # Memory efficiency
                del inp, out, val, pol, fft

    # Computes the quaternion of the triaxial gyrometer
    def build_quaternions(self):

        for knd in ['t', 'v']:

            # Prepares the data to be computed
            with h5py.File(self.path, 'r') as dtb: 
                shp = dtb['gyr_x_{}'.format(knd)].shape
                val = np.empty((shp[0], 3, shp[1]))
                for idx, ele in enumerate(['x', 'y', 'z']):
                    val[:,idx,:] = dtb['gyr_{}_{}'.format(ele, knd)].value

            # Multiprocessed computation
            pol = multiprocessing.Pool(processes=self.njobs)
            qua = np.asarray(pol.map(compute_quaternion, val))
            pol.close()
            pol.join()
            # Serialize the output
            with h5py.File(self.path, 'a') as dtb :
                for idx in np.arange(qua.shape[1]):
                    out = 'qua_{}_{}'.format(idx, knd)
                    if dtb.get(out): del dtb[out]
                    dtb.create_dataset(out, data=qua[:,idx,:])

            # Memory efficiency
            del out, val, shp, pol, qua

    # Built the betti curves out of the norms of the signals
    def build_betti_curves(self):

        for typ in ['acc', 'gyr']:
            for knd in ['t', 'v']:

                inp = 'n_{}_{}'.format(typ, knd)
                
                with h5py.File(self.path, 'r') as dtb: val = dtb[inp].value
                # Multiprocessed computation
                pol = multiprocessing.Pool(processes=self.njobs)
                tda = np.asarray(pol.map(compute_betti_curves, val))
                pol.close()
                pol.join()
                # Serialize the output
                with h5py.File(self.path, 'a') as dtb :
                    out = 'bup_{}_{}'.format(typ[0], knd)
                    if dtb.get(out): del dtb[out]
                    dtb.create_dataset(out, data=tda[:,0,:])
                    out = 'bdw_{}_{}'.format(typ[0], knd)
                    if dtb.get(out): del dtb[out]
                    dtb.create_dataset(out, data=tda[:,1,:])
                # Memory efficiency
                del inp, out, val, pol, tda

    # Preprocess the raw signals
    # out_dtb refers to the output database
    def standardize(self, out_dtb):

        with h5py.File(self.path, 'r') as dtb:
            lst = remove_doublon([key[:-2] for key in list(dtb.keys()) if key[-1] == 't'])
            sze = dtb['label_t'].shape[0]

        for key in tqdm.tqdm(lst):

            if key == 'label':
                for typ in ['_t', '_v']:
                    with h5py.File(out_dtb, 'a') as out:
                        if out.get(key + typ): del out[key + typ]
                        with h5py.File(self.path, 'r') as dtb:
                            out.create_dataset(key + typ, data=dtb[key + typ].value)

            elif key in ['fea', 'fft_a', 'fft_g']:
                # Build the scaler
                mms = MinMaxScaler(feature_range=(-1,1))
                sts = StandardScaler(with_std=False)
                pip = Pipeline([('mms', mms), ('sts', sts)])
                # Stack the data for scaling
                with h5py.File(self.path, 'r') as dtb:
                    val = np.vstack((dtb[key + '_t'].value, dtb[key + '_v'].value))
                    val = pip.fit_transform(val)
                with h5py.File(out_dtb, 'a') as out:
                    if out.get(key + '_t'): del out[key + '_t']
                    out.create_dataset(key + '_t', data=val[:sze])
                    if out.get(key + '_v'): del out[key + '_v']
                    out.create_dataset(key + '_v', data=val[sze:])
                # Memory efficiency
                del mms, sts, pip, val

            else:
                # Build the scaler
                mms = MinMaxScaler(feature_range=(0, 2))
                sts = StandardScaler(with_std=False)
                pip = Pipeline([('mms', mms), ('sts', sts)])
                # Stack the data for scaling
                with h5py.File(self.path, 'a') as dtb:
                    val = np.vstack((dtb[key + '_t'].value, dtb[key + '_v'].value))
                    val = pip.fit_transform(np.hstack(val).reshape(-1,1)).reshape(val.shape)
                with h5py.File(out_dtb, 'a') as out:
                    if out.get(key + '_t'): del out[key + '_t']
                    out.create_dataset(key + '_t', data=val[:sze])
                    if out.get(key + '_v'): del out[key + '_v']
                    out.create_dataset(key + '_v', data=val[sze:])
                # Memory efficiency
                del mms, sts, pip, val

    # Build the datasets for cross-validation
    # inp_dtb points to the scaled database
    # folds refers to the amount of cross-validation rounds
    # msk_labels refers to whether mask data or not
    def cross_validation(self, inp_dtb, folds, msk_transitions=False):

        if msk_transitions: msk_labels = np.arange(6, 12)
        else: msk_labels = []

        # Prepares the mask relative to the labels
        with h5py.File(inp_dtb, 'r') as dtb: 
            lab = dtb['label_t'].value
            msk = get_mask(lab, lab_to_del=msk_labels)
            lab = lab[msk]
            l_v = dtb['label_v'].value
            m_v = get_mask(l_v, lab_to_del=msk_labels)

        # Defines the cross-validation splits
        kfs = StratifiedKFold(n_splits=folds, shuffle=True)

        # For each round, creates a new dataset
        for idx, (i_t, i_e) in enumerate(kfs.split(lab, lab)):

            if msk_transitions: output = './dataset/CV_ITER_MSK_{}.h5'.format(idx)
            else: output = './dataset/CV_ITER_{}.h5'.format(idx)            
            print('\n# Building {}'.format(output))

            # Split the training set into both training and testing
            with h5py.File(inp_dtb, 'r') as dtb:

                print('# Train and Test ...')
                time.sleep(0.5)
                lst = [key[:-2] for key in dtb.keys() if key[-1] == 't']
                lsv = [key for key in dtb.keys() if key[-1] == 'v']

                for key in tqdm.tqdm(lst):

                    with h5py.File(output, 'a') as out:

                        key_t = '{}_t'.format(key)
                        key_e = '{}_e'.format(key)
                        val = dtb[key_t].value[msk]

                        if out.get(key_t): del out[key_t]
                        out.create_dataset(key_t, data=val[i_t])
                        if dtb.get(key_e): del out[key_e]
                        out.create_dataset(key_e, data=val[i_e])
                        del val

                for key in tqdm.tqdm(lsv):

                    with h5py.File(output, 'a') as out:

                        if out.get(key): del out[key]
                        out.create_dataset(key, data=dtb[key].value[m_v])
