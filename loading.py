# Author : DINDIN Meryll
# Date : 01/11/2017

from toolbox import *

# Build a specific loader for the HAPT Dataset matching features with their raw signals
class HAPT_Loader :

    # Initialization
    def __init__(self, path, fea_path='./Fea_Data', raw_path='./Raw_Data', max_jobs=multiprocessing.cpu_count()) :

        # Cares about multiprocessing instances
        self.njobs = max_jobs
        # Root paths
        self.fea_path = fea_path
        self.raw_path = raw_path
        # Represents the 30% of validation subset
        self.usr_train = np.unique(read_text_file('{}/subject_id_train.txt'.format(self.fea_path), 'Subjects').values)
        self.usr_valid = np.unique(read_text_file('{}/subject_id_test.txt'.format(self.fea_path), 'Subjects').values)
        # Represents the other 70%
        self.usr_train = [ele for ele in range(1, 31) if ele not in self.usr_valid]
        # Defines conditions relative to the experiment
        self.time_window = 128
        self.overlap_rto = 0.5
        # Path for serialization
        self.path = path

    # Load the features relative to the signals
    def load_fea(self) :

        # Load the features names
        with open('{}/features.txt'.format(self.fea_path)) as raw : lab = raw.readlines()
        for ind in range(len(lab)) : 
            tmp = str(lab[ind].replace('\n','').replace(' ',''))
            if tmp in lab : tmp = tmp.replace('1', '2')
            if tmp in lab : tmp = tmp.replace('2', '3')
            lab[ind] = tmp
        print('!!! Labels have been corrected ...')
        # Training set
        X_tr = pd.read_csv('{}/X_train.txt'.format(self.fea_path), sep='\n', delimiter=' ', header=None, keep_default_na=False, dtype=np.float32)
        X_tr.columns = lab
        l_tr = read_text_file('{}/y_train.txt'.format(self.fea_path), 'Labels')
        i_tr = read_text_file('{}/subject_id_train.txt'.format(self.fea_path), 'Subjects')
        # Validation set
        X_va = pd.read_csv('{}/X_test.txt'.format(self.fea_path), sep='\n', delimiter=' ', header=None, keep_default_na=False, dtype=np.float32)
        X_va.columns = lab
        l_va = read_text_file('{}/y_test.txt'.format(self.fea_path), 'Labels')
        i_va = read_text_file('{}/subject_id_test.txt'.format(self.fea_path), 'Subjects')
        # Save as attribute
        self.train = fast_concatenate([X_tr, l_tr, i_tr], axis=1)
        self.valid = fast_concatenate([X_va, l_va, i_va], axis=1)
        # Memory efficiency
        del X_va, l_va, i_va, lab, X_tr, l_tr, i_tr, raw
        # Serialize the features in database
        with h5py.File(self.path, 'w') as dtb :
            dtb.create_dataset('FEA_t', data=remove_columns(self.train, ['Subjects', 'Labels']))
            print(' -> Training features serialized ...')
            dtb.create_dataset('FEA_e', data=remove_columns(self.valid, ['Subjects', 'Labels']))
            print(' -> Validation features serialized ...')
        # Memory efficiency
        del self.train, self.valid
        print('|-> Features serialized ...')

    # Loads the raw signals as dataframe
    def load_signals(self) :

        # Where to gather the constructed dataframes
        raw = []
        # Extracts iteratively
        for fle in remove_doublon(['_'.join(fle.split('_')[1:]) for fle in os.listdir(self.raw_path)]) :
            try : 
                # Load the accelerometer data
                acc = pd.read_csv('{}/acc_{}'.format(self.raw_path, fle), sep='\n', delimiter=' ', header=None, keep_default_na=False, dtype=np.float32)
                acc.columns = ['Acc_x', 'Acc_y', 'Acc_z']
                # Load the gyrometer data
                gyr = pd.read_csv('{}/gyro_{}'.format(self.raw_path, fle), sep='\n', delimiter=' ', header=None, keep_default_na=False, dtype=np.float32)
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
        lab = pd.read_csv('{}/labels.txt'.format(self.raw_path), sep='\n', delimiter=' ', header=None)
        lab.columns = ['Experience', 'User', 'Label', 'Begin', 'End']
        # Save as attributes
        self.raw_signals = raw
        self.description = lab
        # Memory efficiency
        del raw, lab

    # Slice the signal accordingly to the time_window and overlap
    def load_raw(self) :

        # First, load the signals
        self.load_signals()

        # Local function for slicing
        def slice_signal(sig) :

            if len(sig) < self.time_window : return []
            else :
                # Init variables
                tme, srt, top = [], 0, len(sig)
                # Prepares multiprocessing
                while srt <= top - self.time_window :
                    tme.append((srt, srt + self.time_window))
                    srt += int(self.overlap_rto * self.time_window)
                # Launch multiprocessing
                pol = multiprocessing.Pool(processes=min(len(tme), self.njobs))
                mvs = pol.map(partial(extract_napt, data=sig), tme)
                pol.close()
                pol.join()
                # Memory efficiency
                del tme, srt, top, pol
                # Return the sliced signals
                return mvs

        # Where to gather the new arrays
        X_tr, y_tr, X_va, y_va = [], [], [], []
        # Deals with the training set
        for ids in self.usr_train :
            print(' -> Getting rid of user {} ...'.format(ids))
            for exp in np.unique(self.description.query('User == {}'.format(ids))['Experience']) :
                cut = self.description.query('Experience == {} & User == {}'.format(exp, ids))
                for val in cut[['Label', 'Begin', 'End']].values :
                    tmp = self.raw_signals.query('Experience == {} & User == {}'.format(exp, ids))
                    sig = slice_signal(remove_columns(tmp[val[1]:val[2]+1], ['Experience', 'User']))
                    y_tr += list(np.full(len(sig), val[0]))
                    X_tr += sig
                    print('    {} samples have been extracted !'.format(len(sig)))
                    del tmp, sig
                del cut
        # Deals with the validation set
        for ids in self.usr_valid :
            print(' -> Getting rid of user {} ...'.format(ids))
            for exp in np.unique(self.description.query('User == {}'.format(ids))['Experience']) :
                cut = self.description.query('Experience == {} & User == {}'.format(exp, ids))
                for val in cut[['Label', 'Begin', 'End']].values :
                    tmp = self.raw_signals.query('Experience == {} & User == {}'.format(exp, ids))
                    sig = slice_signal(remove_columns(tmp[val[1]:val[2]+1], ['Experience', 'User']))
                    y_va += list(np.full(len(sig), val[0]))
                    X_va += sig
                    print('    {} samples have been extracted !'.format(len(sig)))
                    del tmp, sig
                del cut
        # Serialize the features in database
        with h5py.File(self.path, 'r+') as dtb :
            dtb.create_dataset('ACC_t', data=np.asarray(X_tr)[:,0:3,:])
            dtb.create_dataset('ACC_e', data=np.asarray(X_va)[:,0:3,:])
            dtb.create_dataset('GYR_t', data=np.asarray(X_tr)[:,3:6,:])
            dtb.create_dataset('GYR_e', data=np.asarray(X_va)[:,3:6,:])
            dtb.create_dataset('N_A_t', data=np.asarray(X_tr)[:,6,:])
            dtb.create_dataset('N_A_e', data=np.asarray(X_va)[:,6,:])
            dtb.create_dataset('N_G_t', data=np.asarray(X_tr)[:,7,:])
            dtb.create_dataset('N_G_e', data=np.asarray(X_va)[:,7,:])
            dtb.create_dataset('y_train', data=np.asarray(y_tr).astype(int) - 1)
            dtb.create_dataset('y_valid', data=np.asarray(y_va).astype(int) - 1)
        print('|-> Signals serialized ...')
        # Memory efficiency
        del X_tr, X_va, y_tr, y_va, self.raw_signals, self.description

# Build a specific loader for the SHL Dataset
class SHL_Loader :

    # Initialization
    def __init__(self, path, time_window, overlap, anatomy) :

        # Serialization path
        self.path = path
        # Attributes relative to slicing
        self.time_window = time_window
        self.overlap = overlap
        self.anatomy = anatomy
        self.njobs = multiprocessing.cpu_count()

    # Load the raw signals as dataframe
    def load_raw(self) :

        # Local function for slicing
        def slice_signal(sig) :

            if len(sig) < self.time_window : return []
            else :
                # Init variables
                tme, srt, top = [], 0, len(sig)
                # Prepares multiprocessing
                while srt <= top - self.time_window :
                    tme.append((srt, srt + self.time_window))
                    srt += int(self.overlap * self.time_window)
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
                print('|-> Dealing with {} : File {}/{}_Motion.txt'.format(usr, dry, self.anatomy))
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
                print('|-> Signal may be split into {} events'.format(len(idx)))
                for ind, ele in enumerate(idx) :
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
            with h5py.File(self.path, 'w') as dtb :
                dtb.create_group(usr)
                dtb[usr].create_dataset('ACC', data=np.asarray(mvs)[:,0:3,:])
                dtb[usr].create_dataset('GYR', data=np.asarray(mvs)[:,3:6,:])
                dtb[usr].create_dataset('N_A', data=np.asarray(mvs)[:,6,:])
                dtb[usr].create_dataset('N_G', data=np.asarray(mvs)[:,7,:])
                dtb[usr].create_dataset('y', data=np.asarray(lbl).astype(int) - 1)
        print('|-> Signals serialized ...')

# Build a way to add features vectors
class Constructor :

    # Initialization
    def __init__(self, path, output, max_jobs=multiprocessing.cpu_count()) :

        # Defines on which database the work will be done
        self.path = path
        # Defines where to serialize the new database
        self.output = output
        # Defines the pool size for multiprocessing
        self.njobs = max_jobs

    # Computes the fft of each normed inertial signal
    def load_fft(self) :

        with h5py.File(self.path, 'r+') as dtb :
            # FFT of the normed accelerometer
            acc = dtb['N_A_t'].value
            pol = multiprocessing.Pool(processes=self.njobs)
            fft = np.asarray(pol.map(multi_fft, acc))
            pol.close()
            pol.join()
            dtb.create_dataset('FFT_A_t', data=fft)
            acc = dtb['N_A_e'].value
            pol = multiprocessing.Pool(processes=self.njobs)
            fft = np.asarray(pol.map(multi_fft, acc))
            pol.close()
            pol.join()
            dtb.create_dataset('FFT_A_e', data=fft)
            # FFT of the normed gyrometer
            gyr = dtb['N_G_t'].value
            pol = multiprocessing.Pool(processes=self.njobs)
            fft = np.asarray(pol.map(multi_fft, gyr))
            pol.close()
            pol.join()
            dtb.create_dataset('FFT_G_t', data=fft)
            gyr = dtb['N_G_e'].value
            pol = multiprocessing.Pool(processes=self.njobs)
            fft = np.asarray(pol.map(multi_fft, gyr))
            pol.close()
            pol.join()
            dtb.create_dataset('FFT_G_e', data=fft)
            # Memory efficiency
            del acc, pol, fft, gyr
        # Log
        print('|-> FFT computed and saved ...')

    # Computes the quaternion of the triaxial gyrometer
    def load_qua(self) :

        with h5py.File(self.path, 'r+') as dtb :
            # Quaternion multiprocessed computation
            gyr = dtb['GYR_t'].value
            pol = multiprocessing.Pool(processes=self.njobs)
            qua = np.asarray(pol.map(compute_quaternion, gyr))
            pol.close()
            pol.join()
            dtb.create_dataset('QUA_t', data=qua)
            gyr = dtb['GYR_e'].value
            pol = multiprocessing.Pool(processes=self.njobs)
            qua = np.asarray(pol.map(compute_quaternion, gyr))
            pol.close()
            pol.join()
            dtb.create_dataset('QUA_e', data=qua)
            # Memory efficiency
            del gyr, pol, qua
        # Log
        print('|-> Quaternions computed and saved ...')

    # Computes the landscapes out of given signals
    def load_relative_ldc(self, dim_1_alpha=True, dim_2_alpha=True, dim_0_dtm=True, dim_1_dtm=True) :
        
        sys.path.append('../Install/2017-10-02-10-19-30_GUDHI_2.0.1/build/cython/')
        import gudhi

        with h5py.File(self.path, 'r+') as dtb :
            # Compute landscapes for acceleration and rotation speed
            for key in ['ACC_t', 'ACC_e', 'GYR_t', 'GYR_e'] :
                val = [ele.transpose() for ele in dtb[key].value]
                pol = multiprocessing.Pool(processes=self.njobs)
                ldc = pol.map(partial(compute_landscapes, dim_1_alpha=dim_1_alpha, dim_2_alpha=dim_2_alpha, dim_0_dtm=dim_0_dtm, dim_1_dtm=dim_1_dtm), val)
                pol.close()
                pol.join()
                dtb.create_dataset('R_L_{}'.format(key), data=ldc)
                del val, pol, ldc
            # Compute landscapes for quaternions
            for key in ['QUA_t', 'QUA_e'] :
                val = [ele.transpose() for ele in dtb[key].value[:,1:4,:]]
                pol = multiprocessing.Pool(processes=self.njobs)
                ldc = pol.map(partial(compute_landscapes, dim_1_alpha=dim_1_alpha, dim_2_alpha=dim_2_alpha, dim_0_dtm=dim_0_dtm, dim_1_dtm=dim_1_dtm), val)
                pol.close()
                pol.join()
                dtb.create_dataset('R_L_{}'.format(key), data=ldc)
                del val, pol, ldc

    # Preprocess the raw signals
    def standardize(self) :

        # Defines the output database
        out = h5py.File(self.output, 'w')
        # Apply shuffling to the data
        with h5py.File(self.path, 'r') as dtb : 
            idt = shuffle(range(dtb['y_train'].shape[0]))
            out.create_dataset('y_train', data=dtb['y_train'].value[idt])
            ide = shuffle(range(dtb['y_valid'].shape[0]))
            out.create_dataset('y_valid', data=dtb['y_valid'].value[ide])
        # Dict where to gather the scalers
        put = dict()
        # Standardize 2D raw signals
        for typ in ['ACC', 'GYR', 'QUA'] :
            try :
                with h5py.File(self.path, 'r') as dtb :
                    # Fitting
                    sze = dtb['{}_t'.format(typ)].shape[1]
                    sca = [Pipeline([('mms', MinMaxScaler(feature_range=(-1,1))), ('std', StandardScaler(with_std=False))]) for i in range(sze)]
                    acc = np.asarray([dtb['{}_t'.format(typ)].value[:,sig,:] for sig in range(sze)])
                    acc = acc.reshape(acc.shape[0], acc.shape[1]*acc.shape[2])
                    for idx in range(3) : acc[idx,:] = sca[idx].fit_transform(acc[idx,:].reshape(-1,1)).reshape(acc.shape[1])
                    acc = np.asarray([acc[idx,:].reshape(dtb['{}_t'.format(typ)].shape[0], int(dtb['{}_t'.format(typ)].shape[1]*dtb['{}_t'.format(typ)].shape[2]/sze)) for idx in range(acc.shape[0])])
                    acc = np.asarray([[acc[idx, ind, :] for idx in range(acc.shape[0])] for ind in range(acc.shape[1])])
                    out.create_dataset('{}_t'.format(typ), data=acc[idt])
                    # Spreading
                    acc = np.asarray([dtb['{}_e'.format(typ)].value[:,sig,:] for sig in range(sze)])
                    acc = acc.reshape(acc.shape[0], acc.shape[1]*acc.shape[2])
                    for idx in range(3) : acc[idx,:] = sca[idx].fit_transform(acc[idx,:].reshape(-1,1)).reshape(acc.shape[1])
                    acc = np.asarray([acc[idx,:].reshape(dtb['{}_e'.format(typ)].shape[0], int(dtb['{}_e'.format(typ)].shape[1]*dtb['{}_e'.format(typ)].shape[2]/sze)) for idx in range(acc.shape[0])])
                    acc = np.asarray([[acc[idx, ind, :] for idx in range(acc.shape[0])] for ind in range(acc.shape[1])])
                    out.create_dataset('{}_e'.format(typ), data=acc[ide])
                    # Memory efficiency
                    put[typ] = sca
                    del sca, acc, sze
                    print('! Multi-axial {} signal scaled ...'.format(typ))
            except :
                print('! No {} key recognized ...')
        # Standardize 1D raw signals, boolean for logarithmic transform
        for typ, log in [('N_A', True), ('N_G', True)] :
            try : 
                with h5py.File(self.path, 'r') as dtb : 
                    # Fitting
                    sca = Pipeline([('mms', MinMaxScaler(feature_range=(-1,1))), ('std', StandardScaler(with_std=False))])
                    if log : n_a = np.log(np.hstack(dtb['{}_t'.format(typ)].value))
                    else : n_a = np.hstack(dtb['{}_t'.format(typ)].value)
                    n_a = sca.fit_transform(n_a.reshape(-1,1)).reshape(n_a.shape[0])
                    n_a = n_a.reshape(dtb['{}_t'.format(typ)].shape[0], dtb['{}_t'.format(typ)].shape[1])
                    out.create_dataset('{}_t'.format(typ), data=n_a[idt])
                    # Spreading
                    n_a = np.log(np.hstack(dtb['{}_e'.format(typ)].value))
                    n_a = sca.transform(n_a.reshape(-1,1)).reshape(n_a.shape[0])
                    n_a = n_a.reshape(dtb['{}_e'.format(typ)].shape[0], dtb['{}_e'.format(typ)].shape[1])
                    out.create_dataset('{}_e'.format(typ), data=n_a[ide])
                    # Memory efficiency
                    put[typ] = sca
                    del sca, n_a
                    print('! Single-axial {} signal scaled ...'.format(typ))
            except :
                print('! No {} key recognized ...')
        # Standardize features
        for typ in ['FEA', 'FFT_A', 'FFT_G'] :
            try : 
                with h5py.File(self.path, 'r') as dtb : 
                    # Fitting
                    sca = Pipeline([('mms', MinMaxScaler(feature_range=(-1,1))), ('std', StandardScaler())])
                    fea = sca.fit_transform(dtb['{}_t'.format(typ)].value)
                    out.create_dataset('{}_t'.format(typ), data=fea[idt])
                    # Spreading
                    fea = sca.transform(dtb['{}_e'.format(typ)].value)
                    out.create_dataset('{}_e'.format(typ), data=fea[ide])
                    # Memory efficiency
                    put[typ] = sca
                    del fea, sca
                    print('! Features {} scaled ...'.format(typ))
            except :
                print('! No {} key recognized ...')
        # Spread the rest of the keys in the new database
        with h5py.File(self.path, 'r') as dtb :
            for key in dtb.keys() :
                if key not in out.keys() :
                    if key[-1] == 't' : out.create_dataset(key, data=dtb[key].value[idt])
                    elif key[-1] == 'e' : out.create_dataset(key, data=dtb[key].value[ide])
                    else : out.create_dataset(key, data=dtb[key].value)
                    print('> Added {} ...'.format(key))
        # Avoid corruption
        out.close()
        # Serialize the resulting scalers
        raw = open('/'.join(self.output.split('/')[:-1]) + '/scalers.pk', 'wb')
        pickle.dump(put, raw)
        raw.close()
        del put, raw

    # Defines a loading instance caring about both features and raw signals
    def build_database(self, landscapes=False, standardize=True) :

        # Load signals
        self.load_fft()
        self.load_qua()
        # Defines a tda boolean
        if landscapes : self.load_relative_ldc()
        # Defines a standardization boolean
        if standardize : self.standardize()
