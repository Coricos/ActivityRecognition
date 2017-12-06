# Author : DINDIN Meryll
# Date : 16/11/2017

# If forced back to CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from toolbox import *

# Deep-learning models of mixed channels
class Creator :
    
    # Initialization
    def __init__(self, path, merge_size=20, with_n_a=True, with_n_g=True, with_acc=True, with_gyr=True, with_fft=False, with_qua=False, with_fea=False, with_R_l=False, with_A_l=False, truncate=False) :

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
        with h5py.File(path, 'r') as dtb :
            # Load the labels and initialize training and testing sets
            self.l_t = dtb['y_train'].value
            self.l_e = dtb['y_valid'].value
        # If masks are necessary
        if truncate : m_t, m_e = get_mask(self.l_t), get_mask(self.l_e)
        else : m_t, m_e = np.ones(len(self.l_t), dtype=bool), np.ones(len(self.l_e), dtype=bool)
        self.l_t = self.l_t[m_t]
        self.l_e = self.l_e[m_e]
        # Load data accordingly
        with h5py.File(path, 'r') as dtb :
            if with_n_a :
                self.n_a_t = reformat(dtb['N_A_t'].value[m_t], '1D')
                self.n_a_e = reformat(dtb['N_A_e'].value[m_e], '1D')
                self.name[0] = 'T'
            self.with_n_a = with_n_a
            if with_n_g :
                self.n_g_t = reformat(dtb['N_G_t'].value[m_t], '1D')
                self.n_g_e = reformat(dtb['N_G_e'].value[m_e], '1D')
                self.name[1] = 'T'
            self.with_n_g = with_n_g
            if with_acc :
                self.acc_t = reformat(dtb['ACC_t'].value[m_t], '2D')
                self.acc_e = reformat(dtb['ACC_e'].value[m_e], '2D')
                self.name[2] = 'T'
            self.with_acc = with_acc
            if with_gyr :
                self.gyr_t = reformat(dtb['GYR_t'].value[m_t], '2D')
                self.gyr_e = reformat(dtb['GYR_e'].value[m_e], '2D')
                self.name[3] = 'T'
            self.with_gyr = with_gyr
            if with_fft :
                self.f_A_t = dtb['FFT_A_t'].value[m_t]
                self.f_A_e = dtb['FFT_A_e'].value[m_e]
                self.f_G_t = dtb['FFT_G_t'].value[m_t]
                self.f_G_e = dtb['FFT_G_e'].value[m_e]
                self.name[4] = 'T'
            self.with_fft = with_fft
            if with_fea :
                self.fea_t = dtb['FEA_t'].value[m_t]
                self.fea_e = dtb['FEA_e'].value[m_e]
                self.name[5] = 'T'
            self.with_fea = with_fea
            if with_qua :
                self.qua_t = reformat(dtb['QUA_t'].value[m_t], '2D')
                self.qua_e = reformat(dtb['QUA_e'].value[m_e], '2D')
                self.name[6] = 'T'
            self.with_qua = with_qua
            if with_R_l :
                self.r_l_acc_t = dtb['R_L_ACC_t'].value[m_t]
                self.r_l_acc_e = dtb['R_L_ACC_e'].value[m_e]
                self.r_l_gyr_t = dtb['R_L_GYR_t'].value[m_t]
                self.r_l_gyr_e = dtb['R_L_GYR_e'].value[m_e]
                self.r_l_qua_t = dtb['R_L_QUA_t'].value[m_t]
                self.r_l_qua_e = dtb['R_L_QUA_e'].value[m_e]
                self.name[7] = 'T'
            self.with_R_l = with_R_l
            if with_A_l :
                self.a_l_acc_t = dtb['A_L_ACC_t'].value[m_t]
                self.a_l_acc_e = dtb['A_L_ACC_e'].value[m_e]
                self.a_l_gyr_t = dtb['A_L_GYR_t'].value[m_t]
                self.a_l_gyr_e = dtb['A_L_GYR_e'].value[m_e]
                self.a_l_qua_t = dtb['A_L_QUA_t'].value[m_t]
                self.a_l_qua_e = dtb['A_L_QUA_e'].value[m_e]
                self.name[8] = 'T'
            self.with_A_l = with_A_l

        # Build real name
        self.name = ''.join(self.name)
        # Avoid corruption
        dtb.close()

    # Add convolution model
    def add_CONV_1D(self, channel, dropout) :

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
        mod = Activation('tanh')(mod)
        mod = MaxPooling1D(pool_size=2)(mod)
        mod = Dropout(dropout)(mod)
        mod = Conv1D(50, 25)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(dropout)(mod)
        mod = Conv1D(50, 10)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(dropout)(mod)
        mod = GlobalMaxPooling1D()(mod)
        mod = Dense(self.merge_size, activation='relu')(mod)

        # Add model to main model
        self.input.append(inp)
        self.merge.append(mod)

    # Add convolution model
    def add_CONV_2D(self, channel, dropout) :

        # Depends on the selected channel
        if channel == 'acc' : 
            inp = Input(shape=self.acc_t[0].shape)
            self.train.append(self.acc_t)
            self.valid.append(self.acc_e)
            mod = Convolution2D(64, (self.acc_t.shape[2], 60), data_format='channels_first')(inp)
            del self.acc_t, self.acc_e
        elif channel == 'gyr' : 
            inp = Input(shape=self.gyr_t[0].shape)
            self.train.append(self.gyr_t)
            self.valid.append(self.gyr_e)
            mod = Convolution2D(64, (self.gyr_t.shape[2], 60), data_format='channels_first')(inp)
            del self.gyr_t, self.gyr_e
        elif channel == 'qua' :
            inp = Input(shape=self.qua_t[0].shape)
            self.train.append(self.qua_t)
            self.valid.append(self.qua_e)
            mod = Convolution2D(64, (self.qua_t.shape[2], 60), data_format='channels_first')(inp)
            del self.qua_t, self.qua_e

        # Build model
        mod = Activation('tanh')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = Dropout(dropout)(mod)
        mod = Convolution2D(128, (1, 30), data_format='channels_first')(mod)
        mod = Activation('tanh')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = Dropout(dropout)(mod)
        mod = GlobalAveragePooling2D()(mod)
        mod = Dense(self.merge_size, activation='relu')(mod)

        # Add layers to the model
        self.input.append(inp)
        self.merge.append(mod)

    # Add dense network for handcrafted features
    def add_DENSE(self, channel, dropout) :

        # Depends on the selected channel
        if channel == 'fea' : 
            inp = Input(shape=(self.fea_t.shape[1], ))
            self.train.append(self.fea_t)
            self.valid.append(self.fea_e)
            del self.fea_t, self.fea_e
        elif channel == 'fft_A' : 
            inp = Input(shape=(self.f_A_t.shape[1], ))
            self.train.append(self.f_A_t)
            self.valid.append(self.f_A_e)
            del self.f_A_t, self.f_A_e
        elif channel == 'fft_G' :
            inp = Input(shape=(self.f_G_t.shape[1], ))
            self.train.append(self.f_G_t)
            self.valid.append(self.f_G_e)
            del self.f_G_t, self.f_G_e

        # Build the model
        mod = Dense(200)(inp)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(dropout)(mod)
        mod = Dense(100)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(dropout)(mod)
        mod = Dense(self.merge_size, activation='relu')(mod)
        # Add layers to model
        self.input.append(inp)
        self.merge.append(mod)

    # Add silhouette layer for landscapes
    def add_SILHOUETTE(self, channel, dropout) :

        def silhouette(inp) :

            mod = SilhouetteLayer(int(inp.shape[-1]))(inp)
            mod = GlobalMaxPooling1D()(mod)
            mod = Dense(100)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('sigmoid')(mod)
            mod = Dropout(dropout)(mod)
            mod = Dense(25, activation='relu')(mod)
    
            return mod

        # Build the inputs
        if channel == 'acc' : 
            inp = [Input(shape=(ele[0].shape)) for ele in [self.r_l_acc_t[:,idx,:,:] for idx in range(4)]]
            self.train += [self.r_l_acc_t[:,idx,:,:] for idx in range(4)]
            self.valid += [self.r_l_acc_e[:,idx,:,:] for idx in range(4)]
        if channel == 'gyr' :
            inp = [Input(shape=(ele[0].shape)) for ele in [self.r_l_gyr_t[:,idx,:,:] for idx in range(4)]]
            self.train += [self.r_l_gyr_t[:,idx,:,:] for idx in range(4)]
            self.valid += [self.r_l_gyr_e[:,idx,:,:] for idx in range(4)]
        if channel == 'qua' :
            inp = [Input(shape=(ele[0].shape)) for ele in [self.r_l_qua_t[:,idx,:,:] for idx in range(4)]]
            self.train += [self.r_l_qua_t[:,idx,:,:] for idx in range(4)]
            self.valid += [self.r_l_qua_e[:,idx,:,:] for idx in range(4)]
        
        # Build the silhouettes
        mrg = [silhouette(ele) for ele in inp]
        # Intermediary model
        mod = concatenate(mrg)
        mod = Dense(int(1.5*self.merge_size))(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(dropout)(mod)
        mod = Dense(self.merge_size, activation='relu')(mod)

        # Adds to attributes
        self.input += inp
        self.merge.append(mod)

    # Build the whole model
    def build(self, dropout=0.5) :

        # Look for what has been given
        if self.with_fea : self.add_DENSE('fea', dropout)
        if self.with_acc : self.add_CONV_2D('acc', dropout)
        if self.with_gyr : self.add_CONV_2D('gyr', dropout)
        if self.with_fft : 
            self.add_DENSE('fft_A', dropout)
            self.add_DENSE('fft_G', dropout)
        if self.with_n_a : self.add_CONV_1D('n_a', dropout)
        if self.with_n_g : self.add_CONV_1D('n_g', dropout)
        if self.with_qua : self.add_CONV_2D('qua', dropout)
        if self.with_R_l : 
            self.add_SILHOUETTE('acc', dropout)
            self.add_SILHOUETTE('gyr', dropout)
            self.add_SILHOUETTE('qua', dropout)

    # Defines a GPU-oriented fit_generator
    def train_generator(self, batch_size=32) :

        ind = 0

        while True :

            if ind + batch_size >= self.train[0].shape[0] : ind = 0
                
            yield([ele[ind : ind+batch_size] for ele in self.train], np_utils.to_categorical(self.l_t[ind : ind+batch_size], num_classes=len(np.unique(self.l_t))))

            ind += batch_size

    # Defines a GPU-oriented fit_generator
    def valid_generator(self, batch_size=32) :

        ind = 0

        while True :

            if ind + batch_size >= self.valid[0].shape[0] : ind = 0

            yield([ele[ind : ind+batch_size] for ele in self.valid], np_utils.to_categorical(self.l_e[ind : ind+batch_size], num_classes=len(np.unique(self.l_t))))

            ind += batch_size

    # Lauch the fit
    def learn(self, dropout=0.5, verbose=1, max_epochs=100) :

        # Build the corresponding model
        self.build(dropout=dropout)
        # Shuffle all the data
        idx, self.l_t = shuffle(range(len(self.l_t)), self.l_t)
        self.train = [ele[idx] for ele in self.train]
        # Gather all the model in one dense network
        model = concatenate(self.merge)
        model = Dense(int(0.5 * self.merge_size * len(self.train)))(model)
        model = BatchNormalization()(model)
        model = Activation('tanh')(model)
        model = Dropout(dropout)(model)
        model = Dense(int(0.25 * self.merge_size * len(self.train)))(model)
        model = BatchNormalization()(model)
        model = Activation('tanh')(model)
        model = GaussianDropout(dropout)(model)
        model = Dense(len(np.unique(self.l_t)), activation='softmax')(model)
        # Compile the modelel
        model = Model(inputs=self.input, outputs=model)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        # Implements the early stopping
        if self.truncate : pth = './Truncates/clf_{}.h5'.format(self.name)
        else : pth = './Classifiers/clf_{}.h5'.format(self.name)            
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        check = ModelCheckpoint(pth, period=3, monitor='val_acc', save_best_only=True, mode='auto', save_weights_only=False)
        # Fit the model
        model.fit_generator(self.train_generator(batch_size=32), verbose=verbose, epochs=max_epochs,
                            steps_per_epoch=self.train[0].shape[0]/32, shuffle=True, callbacks=[early, check],
                            validation_data=self.valid_generator(batch_size=32), validation_steps=self.valid[0].shape[0]/32,
                            class_weight=class_weight(self.l_t))
        # Save model as attribute
        model.save(pth)
        self.model = model
        # Memory efficiency
        del self.train, self.l_t, self.merge, self.input, early, check, model
        # Returns the object
        return self

    # Observe its performance
    def evaluate(self) :

        # Compute the predictions
        prd = [np.argmax(pbs) for pbs in self.model.predict(self.valid)]
        # Returns the corresponding dataframe
        return score_verbose(self.l_e, prd)
