# Author : DINDIN Meryll
# Date : 01/11/2017

from toolbox import *

# Build tool to match features with reconstituted raw signals
class Loader :

    # Initialization
    def __init__(self) :

        # Root paths
        self.raw_path = './Raw_Data/'
        self.fea_path = './Fea_Data/'
        # Represents the 30% of validation subset
        self.usr_valid = [2, 4, 10, 12, 13, 18, 20, 24]
        # Represents the other 70%
        self.usr_train = [ele for ele in range(1, 30) if ele not in self.usr_valid]
        # Defines conditions relative to the experiment
        self.time_window = 128
        self.overlap_rto = 0.5

    # Loads the raw signals as dataframe
    def load_signals(self) :

        # Where to gather the constructed dataframes
        raw = []
        # Extracts iteratively
        for fle in tqdm.tqdm(remove_doublon(['_'.join(fle.split('_')[1:]) for fle in os.listdir(self.raw_path)])) :
            # Load the accelerometer data
            acc = pd.read_csv('.{}acc_{}'.format(self.raw_path, fle), sep='\n', delimiter=' ', header=None, keep_default_na=False, dtype=np.float32)
            acc.columns = ['Acc_x', 'Acc_y', 'Acc_z']
            # Load the gyrometer data
            gyr = pd.read_csv('.{}gyro_{}'.format(self.raw_path, fle), sep='\n', delimiter=' ', header=None, keep_default_na=False, dtype=np.float32)
            gyr.columns = ['Gyr_x', 'Gyr_y', 'Gyr_z']
            # Load the metadata
            exp = pd.DataFrame(np.asarray([int(fle.split('exp')[1][:2]) for i in range(len(acc))]), columns=['Experience'])
            usr = pd.DataFrame(np.asarray([int(fle.split('user')[1][:2]) for i in range(len(acc))]), columns=['User'])
            # Build the dataframe
            raw.append(fast_concatenate([exp, usr, acc, gyr], axis=1))
            # Memory efficiency
            del acc, gyr, exp, usr
        # Concatenate every obtained dataframe
        raw = fast_concatenate(raw, axis=0)
        # Build the norms (referential independance)
        raw['Normed_A'] = np.sqrt(np.square(raw['Acc_x'].values) + np.square(raw['Acc_y']) + np.square(raw['Acc_z']))
        raw['Normed_G'] = np.sqrt(np.square(raw['Acc_x'].values) + np.square(raw['Acc_y']) + np.square(raw['Acc_z']))
        # Build the labels
        lab = pd.read_csv('./labels.txt', sep='\n', delimiter=' ', header=None)
        lab.columns = ['Experience', 'User', 'Label', 'Begin', 'End']
        # Save as attributes
        self.raw_signals = raw
        self.description = lab
        # Memory efficiency
        del raw, lab

    # Load the features relative to the signals
    def load_fea(self) :

        # Read labels and users for features
        def read_text_file(path, column) :

            with open('.{}{}'.format(self.fea_path, path), 'r') as raw : res = raw.readlines()
            for ind in range(len(res)) : res[ind] = res[ind].replace('\n', '')

            return pd.DataFrame(np.asarray(res).astype(int), columns=[column])

        # Training set
        X_tr = pd.read_csv('.{}X_train.txt'.format(self.fea_path), sep='\n', delimiter=' ', header=None, keep_default_na=False, dtype=np.float32)
        l_tr = read_text_file('.{}y_train.txt'.format(self.fea_path), 'Labels')
        i_tr = read_text_file('.{}subject_id_train.txt'.format(self.fea_path), 'Subjects')
        # Save as attribute
        self.train = fast_concatenate([X_tr, l_tr, i_tr], axis=1)
        # Memory efficiency
        del X_tr, l_tr, i_tr
        # Validation set
        X_va = pd.read_csv('.{}X_train.txt'.format(self.fea_path), sep='\n', delimiter=' ', header=None, keep_default_na=False, dtype=np.float32)
        l_va = read_text_file('.{}y_train.txt'.format(self.fea_path), 'Labels')
        i_va = read_text_file('.{}subject_id_train.txt'.format(self.fea_path), 'Subjects')
        # Save as attribute
        self.valid = fast_concatenate([X_va, l_va, i_va], axis=1)
        # Memory efficiency
        del X_va, l_va, i_va