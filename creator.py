# Author : DINDIN Meryll
# Date : 16/11/2017

from models import *
from topology import *

# Deep-learning models of mixed channels
class Creator :
    
    # Initialization
    def __init__(self, merge_size=50, with_n_a=True, with_n_g=True, with_acc=True, with_gyr=True, with_fft=False, with_tda=False, with_lds=False, with_hdf=False, with_fea=False, truncate=False) :

        self.njobs = multiprocessing.cpu_count()
        self.merge_size = merge_size
        # Initialize the constructor
        self.input = []
        self.merge = []
        self.train = []
        self.valid = []
        # Default arguments for convolution
        self.truncate = truncate
        # Name
        self.name = ['F' for i in range(9)]
        # Load dataset
        dtb = h5py.File('data.h5', 'r')
        # Load the labels and initialize training and testing sets
        self.l_t = dtb['LAB_t'].value
        self.l_e = dtb['LAB_e'].value
        # If masks are necessary
        if truncate : m_t, m_e = get_mask(self.l_t), get_mask(self.l_e)
        else : m_t, m_e = np.ones(len(self.l_t), dtype=bool), np.ones(len(self.l_e), dtype=bool)
        self.l_t = self.l_t[m_t]
        self.l_e = self.l_e[m_e]
        # Load data accordingly
        if with_n_a or with_n_g or with_acc or with_gyr or with_fft: 
            raw_t, raw_e = dtb['RAW_t'].value, dtb['RAW_e'].value
        if with_n_a :
            self.n_a_t = reformat(raw_t[:,6,:][m_t], '1D')
            self.n_a_e = reformat(raw_e[:,6,:][m_e], '1D')
            self.with_n_a = with_n_a
            self.name[0] = 'T'
        if with_n_g :
            self.n_a_t = reformat(raw_t[:,7,:][m_t], '1D')
            self.n_a_e = reformat(raw_e[:,7,:][m_e], '1D')
            self.with_n_g = with_n_g
            self.name[1] = 'T'
        if with_acc :
            self.n_a_t = reformat(raw_t[:,0:3,:][m_t], '2D')
            self.n_a_e = reformat(raw_e[:,0:3,:][m_e], '2D')
            self.with_acc = with_acc
            self.name[2] = 'T'
        if with_gyr :
            self.n_a_t = reformat(raw_t[:,3:6,:][m_t], '2D')
            self.n_a_e = reformat(raw_e[:,3:6,:][m_e], '2D')
            self.with_gyr = with_gyr
            self.name[3] = 'T'
        if with_fft :
            pol = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            self.fft_t = np.asarray(pol.map(multi_fft, list(raw_t[:,6,:][m_t])))
            self.fft_e = np.asarray(pol.mpa(multi_fft, list(raw_e[:,6,:][m_e])))
            pol.close()
            pol.join()
            del pol
            self.with_fft = with_fft
            self.name[4] = 'T'
        if with_hdf :
            self.hdf_t = dtb['HDF_t'].value[m_t]
            self.hdf_e = dtb['HDF_e'].value[m_e]
            self.with_hdf = with_hdf
            self.name[5] = 'T'
        if with_fea :
            self.fea_t = dtb['FEA_t'].value[m_t]
            self.fea_e = dtb['FEA_e'].value[m_e]
            self.with_fea = with_fea
            self.name[6] = 'T'
        if with_tda :
            self.tda_t = np.hstack((dtb['TDA_A_t'].value[m_t], dtb['TDA_G_t'].value[m_t], dtb['TDA_AG_t'].value[m_t]))
            self.tda_e = np.hstack((dtb['TDA_A_e'].value[m_t], dtb['TDA_G_e'].value[m_t], dtb['TDA_AG_e'].value[m_t])) 
            self.with_tda = with_tda
            self.name[7] = 'T'
        if with_lds :
            self.lA0_t = dtb['LDS_A_0_t'].value[m_t]
            self.lA1_t = dtb['LDS_A_1_t'].value[m_t]
            self.lG0_t = dtb['LDS_G_0_t'].value[m_t]
            self.lG1_t = dtb['LDS_G_1_t'].value[m_t]
            self.AG0_t = dtb['LDS_AG_0_t'].value[m_t]
            self.AG1_t = dtb['LDS_AG_1_t'].value[m_t]
            self.lA0_e = dtb['LDS_A_0_e'].value[m_e]
            self.lA1_e = dtb['LDS_A_1_e'].value[m_e]
            self.lG0_e = dtb['LDS_G_0_e'].value[m_e]
            self.lG1_e = dtb['LDS_G_1_e'].value[m_e]
            self.AG0_e = dtb['LDS_AG_0_e'].value[m_e]
            self.AG1_e = dtb['LDS_AG_1_e'].value[m_e]
            self.with_lds = with_lds
            self.name[8] = 'T'
        # Memory efficiency
        if with_n_a or with_n_g or with_acc or with_gyr or with_fft : 
            del raw_t, raw_e
        # Build real name
        self.name = ''.join(self.name)
        # Avoid corruption
        dtb.close()

    # Add convolution model
    def add_CONV_1D(self, channel) :

        # Depends on the selected channel
        if channel == 'n_a' : 
            inp = Input(shape=self.n_a_t[0].shape)
            self.train.append(self.n_a_t)
            self.valid.append(self.n_a_e)
            del self.n_a_t, self.n_a_e
        elif channel == 'n_g' : 
            inp = Input(shape=self.n_g_t[0].shape)
            self.train.append(self.n_g_t)
            self.valid.append(self.n_g_e)
            del self.n_g_t, self.n_g_e

        # Build the selected model
        mod = Conv1D(100, 50)(inp)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = MaxPooling1D(pool_size=2)(mod)
        mod = Dropout(0.30)(mod)
        mod = Conv1D(50, 25)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = Dropout(0.30)(mod)
        mod = Conv1D(50, 10)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = Dropout(0.30)(mod)
        mod = GlobalMaxPooling1D()(mod)
        mod = Dense(self.merge_size, activation='softmax')(mod)

        # Add model to main model
        self.input.append(inp)
        self.merge.append(mod)

    # Add convolution model
    def add_CONV_2D(self, channel) :

        # Depends on the selected channel
        if channel == 'acc' : 
            inp = Input(shape=self.acc_t[0].shape)
            self.train.append(self.acc_t)
            self.valid.append(self.acc_e)
            del self.acc_t, self.acc_e
        elif channel == 'gyr' : 
            inp = Input(shape=self.gyr_t[0].shape)
            self.train.append(self.gyr_t)
            self.valid.append(self.gyr_e)
            del self.gyr_t, self.gyr_e

        # Build model
        mod = Convolution2D(64, (3, 60), data_format='channels_first')(inp)
        mod = Activation('relu')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = Dropout(0.25)(mod)
        mod = Convolution2D(128, (1, 30), data_format='channels_first')(mod)
        mod = Activation('relu')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = Dropout(0.25)(mod)
        mod = GlobalAveragePooling2D()(mod)
        mod = Dense(self.merge_size, activation='softmax')(mod)

        # Add layers to the model
        self.input.append(inp)
        self.merge.append(mod)

    # Add dense network for handcrafted features
    def add_DENSE(self, channel) :

        # Depends on the selected channel
        if channel == 'fea' : 
            inp = Input(shape=(self.fea_t.shape[1], ))
            self.train.append(self.fea_t)
            self.valid.append(self.fea_e)
            del self.fea_t, self.fea_e
        elif channel == 'hdf' : 
            inp = Input(shape=(self.hdf_t.shape[1], ))
            self.train.append(self.hdf_t)
            self.valid.append(self.hdf_e)
            del self.hdf_t, self.hdf_e
        elif channel == 'fft' : 
            inp = Input(shape=(self.fft_t.shape[1], ))
            self.train.append(self.fft_t)
            self.valid.append(self.fft_e)
            del self.fft_t, self.fft_e
        elif channel == 'tda' :
            inp = Input(shape=(self.tda_t.shape[1], ))
            self.train.append(self.tda_t)
            self.valid.append(self.tda_e)
            del self.tda_t, self.tda_e

        # Build the model
        mod = Dense(200)(inp)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.25)(mod)
        mod = Dense(100)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.25)(mod)
        mod = Dense(self.merge_size, activation='softmax')(mod)
        # Add layers to model
        self.input.append(inp)
        self.merge.append(mod)

    # Add silhouette layer for landscapes
    def add_SILHOUETTE(self) :

        def silhouette(inp) :

            mod = SilhouetteLayer(int(inp.shape[-1]))(inp)
            mod = BatchNormalization()(mod)
            mod = Activation('relu')(mod)
            mod = Dropout(0.25)(mod)
            mod = MaxPooling1D(pool_size=2)(mod)
            mod = GlobalMaxPooling1D()(mod)
            mod = Dense(self.merge_size, activation='softmax')(mod)
    
            return mod

        # Build model
        inp = [Input(shape=(ele[0].shape)) for ele in [self.lA0_t, self.lA1_t, self.lG0_t, self.lG1_t, self.AG0_t, self.AG1_t]]
        mrg = [silhouette(ele) for ele in inp]

        # Intermediary model
        mod = merge(mrg)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.3)(mod)
        mod = Dense(100)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.3)(mod)
        mod = Dense(self.size_merge, activation='softmax')(mod)

        # Adds to attributes
        self.input += inp
        self.merge.append(mod)
        self.train += [self.lA0_t, self.lA1_t, self.lG0_t, self.lG1_t, self.AG0_t, self.AG1_t]
        self.valid += [self.lA0_e, self.lA1_e, self.lG0_e, self.lG1_e, self.AG0_e, self.AG1_e]

    # Build the whole model
    def build(self) :

        # Look for what has been given
        if self.with_fea : self.add_DENSE('fea')
        if self.with_acc : self.add_CONV_2D('acc')
        if self.with_gyr : self.add_CONV_2D('gyr')
        if self.with_fft : self.add_DENSE('fft')
        if self.with_n_a : self.add_CONV_1D('n_a')
        if self.with_n_g : self.add_CONV_1D('n_g')
        if self.with_tda : self.add_DENSE('tda')
        if self.with_lds : self.add_SILHOUETTE()

    # Lauch the fit
    def learn(self, verbose=1, max_epochs=100) :

        # Build the corresponding model
        self.build()
        # Gather all the model in one dense network
        model = merge(self.merge)
        model = BatchNormalization()(model)
        model = Activation('tanh')(model)
        model = Dropout(0.3)(model)
        model = Dense(100)(model)
        model = BatchNormalization()(model)
        model = Activation('tanh')(model)
        model = Dropout(0.3)(model)
        model = Dense(100)(model)
        model = BatchNormalization()(model)
        model = Activation('tanh')(model)
        model = GaussianDropout(0.3)(model)
        model = Dense(len(np.unique(self.l_t)), activation='softmax')(model)
        # Compile the modelel
        model = Model(inputs=self.input, outputs=model)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        # Implements the early stopping
        if self.truncate : 
            pth = '../Truncates/clf_{}_{}_{}s.h5'.format(self.name, self.anatomy, self.time_window)
        else : 
            pth = '../Classifiers/clf_{}_{}_{}s.h5'.format(self.name, self.anatomy, self.time_window)            
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        check = ModelCheckpoint(pth, period=3, monitor='val_acc', save_best_only=True, mode='auto', save_weights_only=False)
        # Fit the model
        model.fit(self.train, np_utils.to_categorical(self.l_t), batch_size=16, epochs=max_epochs, 
                  verbose=verbose, validation_split=0.1, shuffle=True, 
                  sample_weight=sample_weight(self.l_t), callbacks=[early, check])
        # Save model as attribute
        model.save(pth)
        self.model = model
        # Memory efficiency
        del self.train, self.l_t, self.merge, self.input, early, check, model

    # Observe its performance
    def evaluate(self) :

        # Compute the predictions
        prd = [np.argmax(pbs) for pbs in self.model.predict(self.valid)]
        # Returns the corresponding dataframe
        return score_verbose(self.l_e, prd)
