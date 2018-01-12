# Author : DINDIN Meryll
# Date : 16/11/2017

from toolbox import *

# Deep-learning models of mixed channels
class DynamicModel :
    
    # Initialization
    def __init__(self, path, args, merge_size=20, msk_labels=[]) :

        self.merge_size = merge_size
        # Initialize the constructor
        self.input = []
        self.merge = []
        self.arg = args
        self.pth = path
        # Labels and their respective masks
        with h5py.File(path, 'r') as dtb :
            # Load the labels and initialize training and testing sets
            self.m_t = get_mask(dtb['y_train'].value, msk_labels)
            self.m_e = get_mask(dtb['y_valid'].value, msk_labels)
            self.l_t = dtb['y_train'].value[self.m_t]
            self.l_e = dtb['y_valid'].value[self.m_e]

    # Add convolution model
    def add_CONV_1D(self, channel, dropout) :

        # Depends on the selected channel
        with h5py.File(self.pth, 'r') as dtb :
            inp = Input(shape=(dtb['{}_t'.format(channel)].shape[1], 1))

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
        with h5py.File(self.pth, 'r') as dtb :
            inp = Input(shape=(1, dtb['{}_t'.format(channel)][0].shape[0], dtb['{}_t'.format(channel)][0].shape[1]))
            mod = Convolution2D(64, (dtb['{}_t'.format(channel)].shape[1], 60), data_format='channels_first')(inp)

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
        with h5py.File(self.pth, 'r') as dtb :
            sze = dtb['{}_t'.format(channel)].shape[1]
            inp = Input(shape=(sze, ))

        # Build the model
        mod = Dense(int(0.75*sze))(inp)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(dropout)(mod)
        mod = Dense(int(0.33*sze))(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(dropout)(mod)
        mod = Dense(self.merge_size, activation='tanh')(mod)
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
        with h5py.File(self.pth, 'r') as dtb : 
            inp = [Input(shape=(dtb[lab][0].shape)) for lab in ['{}_{}_t'.format(channel, idx) for idx in range(4)]]
        
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

        # Build the model given the arguments
        for key in sorted(self.arg.keys()) :
            # Define the concerned channel
            channel = '_'.join(key.split('_')[1:])
            # Add channel and input to general model
            if self.arg[key][0] and self.arg[key][1] == 'DENSE' :
                self.add_DENSE(channel, dropout)
            if self.arg[key][0] and self.arg[key][1] == 'CONV_1D' :
                self.add_CONV_1D(channel, dropout)
            if self.arg[key][0] and self.arg[key][1] == 'CONV_2D' :
                self.add_CONV_2D(channel, dropout)
            if self.arg[key][0] and self.arg[key][1] == 'SILHOUETTE' :
                self.add_SILHOUETTE(channel, dropout)

    # Defines a GPU-oriented fit_generator
    def train_generator(self, batch_size=32) :

        # Initialization
        ind = 0
        # Infinite generator
        while True :
            # Reinitialize when going too far
            if ind + batch_size >= len(np.where(self.m_t == True)[0]) : ind = 0
            # Initialization of data vector
            vec = []
            # Creating batch
            for key in sorted(self.arg.keys()) :
                # Define the concerned channel
                channel, i_t = '_'.join(key.split('_')[1:]), np.where(self.m_t == True)[0]
                # Adding part to vector
                with h5py.File(self.pth, 'r') as dtb :
                    if self.arg[key][0] and self.arg[key][1] == 'DENSE' :
                        vec.append(dtb['{}_t'.format(channel)][list(i_t[ind:ind+batch_size])])
                    if self.arg[key][0] and self.arg[key][1] == 'CONV_1D' :
                        vec.append(reformat(dtb['{}_t'.format(channel)][list(i_t[ind:ind+batch_size])], '1D'))
                    if self.arg[key][0] and self.arg[key][1] == 'CONV_2D' :
                        vec.append(reformat(dtb['{}_t'.format(channel)][list(i_t[ind:ind+batch_size])], '2D'))
                    if self.arg[key][0] and self.arg[key][1] == 'SILHOUETTE' :
                        for idx in range(4) :
                            vec.append(dtb['{}_{}_t'.format(channel, idx)][list(i_t[ind:ind+batch_size])])
            # Yield the resulting vector
            yield(vec, np_utils.to_categorical(self.l_t[ind:ind+batch_size], num_classes=len(np.unique(self.l_t))))
            # Increments over the batch
            ind += batch_size

    # Defines a GPU-oriented fit_generator
    def valid_generator(self, batch_size=32) :

        # Initialization
        ind = 0
        # Infinite generator
        while True :
            # Reinitialize when going too far
            if ind + batch_size >= len(np.where(self.m_e == True)[0]) : ind = 0
            # Initialization of data vector
            vec = []
            # Creating batch
            for key in sorted(self.arg.keys()) :
                # Define the concerned channel
                channel, i_e = '_'.join(key.split('_')[1:]), np.where(self.m_e == True)[0]
                # Adding part to vector
                with h5py.File(self.pth, 'r') as dtb :
                    if self.arg[key][0] and self.arg[key][1] == 'DENSE' :
                        vec.append(dtb['{}_e'.format(channel)][list(i_e[ind:ind+batch_size])])
                    if self.arg[key][0] and self.arg[key][1] == 'CONV_1D' :
                        vec.append(reformat(dtb['{}_e'.format(channel)][list(i_e[ind:ind+batch_size])], '1D'))
                    if self.arg[key][0] and self.arg[key][1] == 'CONV_2D' :
                        vec.append(reformat(dtb['{}_e'.format(channel)][list(i_e[ind:ind+batch_size])], '2D'))
                    if self.arg[key][0] and self.arg[key][1] == 'SILHOUETTE' :
                        for idx in range(4) :
                            vec.append(dtb['{}_{}_e'.format(channel, idx)][list(i_e[ind:ind+batch_size])])
            # Yield the resulting vector
            yield(vec, np_utils.to_categorical(self.l_e[ind:ind+batch_size], num_classes=len(np.unique(self.l_t))))
            # Increments over the batch
            ind += batch_size

    # Lauch the fit
    def learn(self, output, dropout=0.5, verbose=1, max_epochs=100) :

        # Build the corresponding model
        self.build(dropout=dropout)
        # Gather all the model in one dense network
        model = concatenate(self.merge)
        model = Dense(int(0.5 * self.merge_size * len(self.input)))(model)
        model = BatchNormalization()(model)
        model = Activation('tanh')(model)
        model = Dropout(dropout)(model)
        model = Dense(int(0.25 * self.merge_size * len(self.input)))(model)
        model = BatchNormalization()(model)
        model = Activation('tanh')(model)
        model = GaussianDropout(dropout)(model)
        model = Dense(len(np.unique(self.l_t)), activation='softmax')(model)
        # Compile the modelel
        model = Model(inputs=self.input, outputs=model)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        # Implements the early stopping    
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        check = ModelCheckpoint(output, period=3, monitor='val_acc', save_best_only=True, mode='auto', save_weights_only=False)
        # Fit the model
        model.fit_generator(self.train_generator(batch_size=32),
                            verbose=verbose, epochs=max_epochs,
                            steps_per_epoch=int(len(np.where(self.m_t == True)[0])/32), 
                            shuffle=True, callbacks=[early, check],
                            validation_data=self.valid_generator(batch_size=32), 
                            validation_steps=int(len(np.where(self.m_e == True)[0])/32),
                            class_weight=class_weight(self.l_t))
        # Save model as attribute
        model.save(output)
        self.model = model
        # Memory efficiency
        del self.l_t, self.merge, self.input, early, check, model
        # Returns the object
        return self

    # Observe its performance
    def evaluate(self) :

        # Compute the predictions
        prd = [np.argmax(pbs) for pbs in self.model.predict(self.valid)]
        # Returns the corresponding dataframe
        return score_verbose(self.l_e, prd)
