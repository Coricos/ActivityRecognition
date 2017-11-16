# Author : DINDIN Meryll
# Date : 16/11/2017

from models import *
from topology import *

# Deep-learning models of mixed channels
class Creator :
    
    # Initialization
    def __init__(self, basis, merge_size=50, with_raw=True, with_tda=True, with_lds=True, with_hdf=True, with_fea=True, truncate=False) :

        self.basis = basis
        self.njobs = multiprocessing.cpu_count()
        self.merge_size = merge_size
        # Initialize the constructor
        self.input = []
        self.merge = []
        self.train = []
        self.valid = []
        # Default arguments for convolution
        self.truncate = truncate
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
        if with_raw :
            self.raw_t = dtb['RAW_t'].value[m_t]
            self.raw_e = dtb['RAW_e'].value[m_e]
            self.with_raw = with_raw
        if with_hdf :
            self.hdf_t = dtb['HDF_t'].value[m_t]
            self.hdf_e = dtb['HDF_e'].value[m_e]
            self.with_hdf = with_hdf
        if with_fea :
            self.fea_t = dtb['FEA_t'].value[m_t]
            self.fea_e = dtb['FEA_e'].value[m_e]
            self.with_fea = with_fea
        if with_tda :
            self.t_A_t = dtb['TDA_A_t'].value[m_t]
            self.t_G_t = dtb['TDA_G_t'].value[m_t]
            self.t_A_e = dtb['TDA_A_e'].value[m_e]
            self.t_G_e = dtb['TDA_G_e'].value[m_e]
            self.with_tda = with_tda
        if with_lds :
            self.lA0_t = dtb['LDS_A_0_t'].value[m_t]
            self.lA1_t = dtb['LDS_A_1_t'].value[m_t]
            self.lG0_t = dtb['LDS_G_0_t'].value[m_t]
            self.lG1_t = dtb['LDS_G_1_t'].value[m_t]
            self.lA0_e = dtb['LDS_A_0_e'].value[m_e]
            self.lA1_e = dtb['LDS_A_1_e'].value[m_e]
            self.lG0_e = dtb['LDS_G_0_e'].value[m_e]
            self.lG1_e = dtb['LDS_G_1_e'].value[m_e]
            self.with_lds = with_lds
        # Avoid corruption
        dtb.close()

    # Add convolution model
    def add_conv_1D(self) :

        def conv_1D(inp) :
            
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

            return mod

        # Reformat the raw signals
        self.raw_t = reformat_vectors(self.raw_t, self.basis)
        self.raw_e = reformat_vectors(self.raw_e, self.basis)
        # Add layers to the model
        inp = [Input(shape=self.raw_t[0][0].shape) for num in range(len(self.raw_t))]
        self.input += inp
        self.merge += [conv_1D(ele) for ele in inp]
        self.train += self.raw_t
        self.valid += self.raw_e

    # Add convolution model
    def add_conv_2D(self) :

        # Reformat the raw signals
        self.raw_t = reformat_vectors(self.raw_t, self.basis)
        self.raw_e = reformat_vectors(self.raw_e, self.basis)
        # Build model
        inp = Input(shape=self.raw_t[0].shape)
        mod = Convolution2D(64, (8, 60), data_format='channels_first')(inp)
        mod = Activation('relu')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = Dropout(0.25)(mod)
        mod = Convolution2D(128, (1, 30), data_format='channels_first')(mod)
        mod = Activation('relu')(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = Dropout(0.25)(mod)
        mod = Flatten()(mod)
        mod = Dense(100)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.25)(mod)
        mod = Dense(self.merge_size, activation='softmax')(mod)
        # Add layers to the model
        self.input.append(inp)
        self.merge.append(mod)
        self.train.append(self.raw_t)
        self.valid.append(self.raw_e)

    # Add LSTM model
    def add_LSTM(self) :

        def LSTM(inp) : 

            mod = LSTM(100, return_sequences=True)(inp)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.30)(mod)
            mod = LSTM(100)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.30)(mod)
            mod = Dense(self.merge_size, activation='relu')(mod)

            return mod

        # Reformat the raw signals
        self.raw_t = reformat_vectors(self.raw_t, self.basis)
        self.raw_e = reformat_vectors(self.raw_e, self.basis)
        # Add layers to the model
        inp = [Input(shape=self.raw_t[0][0].shape) for num in range(len(self.raw_t))]
        self.input += inp
        self.merge += [LSTM(ele) for ele in inp]
        self.train += self.raw_t
        self.valid += self.raw_e

    # Add dense network for handcrafted features
    def add_dense_hdf(self) :

        # Build model
        inp = Input(shape=(self.hdf_t.shape[1],))
        mod = Dense(300)(inp)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.25)(mod)
        mod = Dense(self.merge_size, activation='relu')(mod)
        # Add layers to model
        self.input.append(inp)
        self.merge.append(mod)
        self.train.append(self.hdf_t)
        self.valid.append(self.hdf_e)

    # Add dense network for provided features
    def add_dense_fea(self) :

        # Build model
        inp = Input(shape=(self.fea_t.shape[1],))
        mod = Dense(300)(inp)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.25)(mod)
        mod = Dense(self.merge_size, activation='relu')(mod)
        # Add layers to model
        self.input.append(inp)
        self.merge.append(mod)
        self.train.append(self.fea_t)
        self.valid.append(self.fea_e)

    # Add dense network for TDA features
    def add_dense_tda(self) :

        # Build model for acceleration
        inp0 = Input(shape=(self.t_A_t.shape[1],))
        mod0 = Dense(80)(inp0)
        mod0 = BatchNormalization()(mod0)
        mod0 = Activation('tanh')(mod0)
        mod0 = Dropout(0.25)(mod0)
        mod0 = Dense(50)(mod0)
        mod0 = BatchNormalization()(mod0)
        mod0 = Activation('tanh')(mod0)
        mod0 = Dropout(0.25)(mod0)
        mod0 = Dense(self.merge_size, activation='tanh')(mod0)
        # Build model for rotation speed
        inp1 = Input(shape=(self.t_G_t.shape[1],))
        mod1 = Dense(80)(inp1)
        mod1 = BatchNormalization()(mod1)
        mod1 = Activation('tanh')(mod1)
        mod1 = Dropout(0.25)(mod1)
        mod1 = Dense(50)(mod1)
        mod1 = BatchNormalization()(mod1)
        mod1 = Activation('tanh')(mod1)
        mod1 = Dropout(0.25)(mod1)
        mod1 = Dense(self.merge_size, activation='tanh')(mod1)
        # Add layers to model
        self.input += [inp0, inp1]
        self.merge += [mod0, mod1]
        self.train += [self.t_A_t, self.t_G_t]
        self.valid += [self.t_A_e, self.t_G_e]

    # Add silhouette layer for landscapes
    def add_silhouette_layers(self) :

        def silhouette(inp) :

            mod = SilhouetteLayer(int(inp.shape[-1]))(inp)
            mod = BatchNormalization()(mod)
            mod = Activation('relu')(mod)
            mod = Dropout(0.25)(mod)
            mod = MaxPooling1D(pool_size=2)(mod)
            mod = GlobalMaxPooling1D()(mod)
            mod = Dense(self.merge_size, activation='tanh')(mod)
    
            return mod

        # Build model
        inp = [Input(shape=(ele[0].shape)) for ele in [self.lA0_t, self.lA1_t, self.lG0_t, self.lG1_t]]
        self.input += inp
        self.merge += [silhouette(ele) for ele in inp]
        self.train += [self.lA0_t, self.lA1_t, self.lG0_t, self.lG1_t]
        self.valid += [self.lA0_e, self.lA1_e, self.lG0_e, self.lG1_e]

    # Build the whole model
    def build(self) :

        if self.with_raw :
            if self.basis == 'Conv1D' : self.add_conv_1D()
            elif self.basis == 'Conv2D' : self.add_conv_2D()
            elif self.basis == 'LSTM' : self.add_LSTM()
        if self.with_hdf : self.add_dense_hdf()
        if self.with_fea : self.add_dense_fea()
        if self.with_tda : self.add_dense_tda()
        if self.with_lds : self.add_silhouette_layers()

        # Return the object
        return self

    # Lauch the fit
    def learn(self, verbose=1, max_epochs=100) :

        # Limit the session
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Build the corresponding model
        self.build()
        # Gather all the model in one dense network
        model = merge(self.merge)
        model = BatchNormalization()(model)
        model = Activation('tanh')(model)
        model = Dropout(0.3)(model)
        model = Dense(200)(model)
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
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        # Fit the model
        model.fit(self.train, np_utils.to_categorical(self.l_t), batch_size=16, epochs=max_epochs, 
                  verbose=verbose, validation_split=0.2, shuffle=True, 
                  sample_weight=sample_weight(self.l_t), callbacks=[early])
        # Save model as attribute
        self.model = model
        # Return the object
        return self

    # Observe its performance
    def evaluate(self) :

        # Compute the predictions
        prd = [np.argmax(pbs) for pbs in self.model.predict(self.valid)]
        # Returns the corresponding dataframe
        return score_verbose(self.l_e, prd)
