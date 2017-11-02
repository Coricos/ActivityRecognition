# Author : DINDIN Meryll
# Date : 01/11/2017

from toolbox import *

# Build tool to match features with reconstituted raw signals
class Loader :

    # Initialization
    def __init__(self, max_jobs=multiprocessing.cpu_count()-1) :

        # Cares about multiprocessing instances
        self.njobs = max_jobs
        # Root paths
        self.raw_path = './Raw_Data'
        self.fea_path = './Fea_Data'
        # Represents the 30% of validation subset
        self.usr_valid = [2, 4, 10, 12, 13, 18, 20, 24]
        # Represents the other 70%
        self.usr_train = [ele for ele in range(1, 30) if ele not in self.usr_valid]
        # Defines conditions relative to the experiment
        self.time_window = 128
        self.overlap_rto = 0.5

    # Load the features relative to the signals
    def load_fea(self) :

        # Read labels and users for features
        def read_text_file(path, column) :

            with open('{}'.format(path), 'r') as raw : res = raw.readlines()
            for ind in range(len(res)) : res[ind] = res[ind].replace('\n', '')

            return pd.DataFrame(np.asarray(res).astype(int), columns=[column])

        # Load the features names
        with open('./Fea_Data/features.txt') as raw : lab = raw.readlines()
        for ind in range(len(lab)) : lab[ind] = lab[ind].replace('\n','').replace(' ','')
        # Training set
        X_tr = pd.read_csv('{}/X_train.txt'.format(self.fea_path), sep='\n', delimiter=' ', header=None, keep_default_na=False, dtype=np.float32)
        X_tr.columns = lab
        l_tr = read_text_file('{}/y_train.txt'.format(self.fea_path), 'Labels')
        i_tr = read_text_file('{}/subject_id_train.txt'.format(self.fea_path), 'Subjects')
        # Save as attribute
        self.train = fast_concatenate([X_tr, l_tr, i_tr], axis=1)
        # Memory efficiency
        del X_tr, l_tr, i_tr, raw
        # Validation set
        X_va = pd.read_csv('{}/X_train.txt'.format(self.fea_path), sep='\n', delimiter=' ', header=None, keep_default_na=False, dtype=np.float32)
        X_va.columns = lab
        l_va = read_text_file('{}/y_train.txt'.format(self.fea_path), 'Labels')
        i_va = read_text_file('{}/subject_id_train.txt'.format(self.fea_path), 'Subjects')
        # Save as attribute
        self.valid = fast_concatenate([X_va, l_va, i_va], axis=1)
        # Memory efficiency
        del X_va, l_va, i_va, lab
        # Return object
        return self

    # Loads the raw signals as dataframe
    def load_signals(self) :

        # Where to gather the constructed dataframes
        raw = []
        # Extracts iteratively
        for fle in tqdm.tqdm(remove_doublon(['_'.join(fle.split('_')[1:]) for fle in os.listdir(self.raw_path)])) :
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
        raw['Normed_G'] = np.sqrt(np.square(raw['Acc_x'].values) + np.square(raw['Acc_y']) + np.square(raw['Acc_z']))
        # Build the labels
        lab = pd.read_csv('{}/labels.txt'.format(self.raw_path), sep='\n', delimiter=' ', header=None)
        lab.columns = ['Experience', 'User', 'Label', 'Begin', 'End']
        # Save as attributes
        self.raw_signals = raw
        self.description = lab
        # Memory efficiency
        del raw, lab

    # Slice the signal accordingly to the time_window and overlap
    def sliding_extraction(self) :

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
                # Take care of the last slice
                try : tme.append((top - self.time_window, top))
                except : pass
                # Launch multiprocessing
                pol = multiprocessing.Pool(processes=min(len(tme), self.njobs))
                mvs = pol.map(partial(extract, data=sig), tme)
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
            for exp in np.unique(self.description.query('User' == {}.format(ids))['Experience']) :
                cut = self.description.query('Experience == {} & User == {}'.format(exp, ids))
                for val in cut[['Label', 'Begin', 'End']].values :
                    tmp = self.raw_signals.query('Experience == {} & User == {}'.format(exp, ids))
                    sig = slice_signal(remove_columns(tmp, ['Experience', 'User']))
                    y_tr += list(np.full(len(sig), cut[0]))
                    X_tr += sig
                    del tmp, sig
                del cut
        # Deals with the validation set
        for ids in self.usr_valid :
            for exp in np.unique(self.description.query('User' == {}.format(ids))['Experience']) :
                cut = self.description.query('Experience == {} & User == {}'.format(exp, ids))
                for val in cut[['Label', 'Begin', 'End']].values :
                    tmp = self.raw_signals.query('Experience == {} & User == {}'.format(exp, ids))
                    sig = slice_signal(remove_columns(tmp, ['Experience', 'User']))
                    y_va += list(np.full(len(sig), cut[0]))
                    X_va += sig
                    del tmp, sig
                del cut
        # Save as attributes
        self.X_tr = np.asarray(X_tr)
        self.y_tr = np.asarray(y_tr).astype(int) - 1
        self.X_va = np.asarray(X_va)
        self.y_va = np.asarray(y_va).astype(int) - 1
        # Memory efficiency
        del X_tr, X_va, y_tr, y_va, self.raw_signals, self.description

    # Preprocess the raw signals
    def load_raw(self) :

        # Prepares the data
        self.load_signals()
        self.sliding_extraction()

        # Local function for processing
        def process(img, scalers) :

            # Save and apply new adapted format
            sz0, sz1 = img.shape[0], img.shape[1]*img.shape[2]/8
            img = np.asarray([img[:, sig, :] for sig in range(img.shape[1])])
            img = img.reshape(img.shape[0], img.shape[1]*img.shape[2])
            # Rescale the data
            for idx in range(img.shape[0]) : 
                if idx in [6, 7] : 
                    img[idx, :] = scalers[idx].fit_transform(np.log(img[idx, :]).reshape(-1, 1)).reshape(img.shape[1])
                else : 
                    img[idx, :] = scalers[idx].fit_transform(img[idx, :].reshape(-1, 1)).reshape(img.shape[1])
            # Reshape as entry
            img = np.asarray([img[idx, :].reshape(sz0, sz1) for idx in range(img.shape[0])])
            img = np.asarray([[img[idx, ind, :] for idx in range(img.shape[0])] for ind in range(img.shape[1])])
            del sz0, sz1

            return img

        # Defines the scalers
        sca = [Pipeline([('mms', MinMaxScaler(feature_range=(-1,1))), ('std', StandardScaler(with_std=False))]) for i in range(8)]
        # Fit the scalers
        img = np.concatenate((self.X_tr, self.X_va))
        img = np.asarray([img[:, sig, :] for sig in range(img.shape[1])])
        img = img.reshape(img.shape[0], img.shape[1]*img.shape[2])
        for idx in range(img.shape[0]) : 
            if idx in [6, 7] : lst[idx].fit(np.log(img[idx, :]).reshape(-1, 1))
            else : lst[idx].fit(img[idx, :].reshape(-1, 1))
        # Save a attributes the preprocessed versions
        self.X_tr = process(self.X_tr, sca)
        self.X_va = process(self.X_va, sca)
        # Memory efficiency
        del img, sca
        # Return object
        return self
