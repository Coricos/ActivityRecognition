# Author : DINDIN Meryll
# Date : 01/11/2017

from loading import *

# Build the models as initialized
class Models :

    # Initialization
    def __init__(self, model, truncate=False, max_jobs=multiprocessing.cpu_count()-1, reduced=False, red_index=[6, 7]) :

        self.name = model
        self.njobs = max_jobs
        # Differentiate cases
        self.case_fea = ['XGBoost', 'RandomForest']
        self.case_raw = ['Conv1D', 'Conv2D', 'LSTM']
        self.case_bth = ['DeepConv1D', 'DeepConv2D', 'DeepLSTM']
        # Default arguments for convolution
        self.reduced = reduced
        self.red_idx = red_index
        self.truncate = truncate
        # Load dataset
        dtb = h5py.File('data.h5', 'r')
        # Load the labels
        self.l_t = dtb['LAB_t'].value
        self.l_e = dtb['LAB_e'].value
        # Load specific intel according to the problematic
        if model in self.case_fea :
            if truncate : 
                [self.h_t, self.f_t], self.l_t = truncate_data([dtb['HDF_t'].value, dtb['FEA_t'].value], self.l_t)
                [self.h_e, self.f_e], self.l_e = truncate_data([dtb['HDF_e'].value, dtb['FEA_e'].value], self.l_e)
            else : 
                self.h_t, self.f_t = dtb['HDF_t'].value, dtb['FEA_t'].value
                self.h_e, self.f_e = dtb['HDF_e'].value, dtb['FEA_e'].value
            print('  ! Fea_Train_Mean : {}, Fea_Valid_Mean : {}'.format(round(np.mean(self.f_t), 3), round(np.mean(self.f_e), 3)))
            print('  ! Hdf_Train_Mean : {}, Hdf_Valid_Mean : {}'.format(round(np.mean(self.h_t), 3), round(np.mean(self.h_e), 3)))

        if model in self.case_raw : 
            if truncate : 
                [self.r_t], self.l_t = truncate_data([dtb['RAW_t'].value], self.l_t)
                [self.r_e], self.l_e = truncate_data([dtb['RAW_e'].value], self.l_e)
            else : 
                self.r_t = dtb['RAW_t'].value
                self.r_e = dtb['RAW_e'].value
            print('  ! Raw_Train_Mean : {}, Raw_Valid_Mean : {}'.format(round(np.mean(self.r_t), 3), round(np.mean(self.r_e), 3)))

        if model in self.case_bth : 
            if truncate :
                [self.h_t, self.f_t, self.r_t], self.l_t = truncate_data([dtb['HDF_t'].value, dtb['FEA_t'].value, dtb['RAW_t'].value], self.l_t)
                [self.h_e, self.f_e, self.r_e], self.l_e = truncate_data([dtb['HDF_e'].value, dtb['FEA_e'].value, dtb['RAW_e'].value], self.l_e)
            else : 
                self.h_t, self.f_t, self.r_t = dtb['HDF_t'].value, dtb['FEA_t'].value, dtb['RAW_t'].value
                self.h_e, self.f_e, self.r_e = dtb['HDF_e'].value, dtb['FEA_e'].value, dtb['RAW_e'].value
            print('  ! Fea_Train_Mean : {}, Fea_Valid_Mean : {}'.format(round(np.mean(self.f_t), 3), round(np.mean(self.f_e), 3)))
            print('  ! Hdf_Train_Mean : {}, Hdf_Valid_Mean : {}'.format(round(np.mean(self.h_t), 3), round(np.mean(self.h_e), 3)))
            print('  ! Raw_Train_Mean : {}, Raw_Valid_Mean : {}'.format(round(np.mean(self.r_t), 3), round(np.mean(self.r_e), 3)))
        # Avoid corruption
        dtb.close()

    # Launch the random searched XGBoost model
    def xgboost(self, n_iter=50, verbose=0) :

        # Prepares the data
        X_tr, y_tr = shuffle(np.hstack((self.f_t, self.h_t)), self.l_t)
        # Defines the model
        clf = xgboost.XGBClassifier(nthread=self.njobs)
        prm = {'learning_rate': [0.01, 0.1, 0.25, 0.5, 0.75, 1.0], 'max_depth': randint(10, 30),
               'n_estimators': randint(250, 300),'gamma': [0.01, 0.001, 0.0001]}
        # Launching the fine-tuning
        clf = RandomizedSearchCV(clf, verbose=verbose, cv=5, param_distributions=prm, n_iter=n_iter, 
                                 scoring=['accuracy', 'neg_log_loss', 'f1_weighted'], refit='accuracy')
        # Log
        print('|-> Learning on {} vectors through cross-validation ...'.format(X_tr.shape[0]))
        # Launch the model
        clf.fit(X_tr, y_tr, sample_weight=sample_weight(y_tr))
        # Log results
        msg = '  ~ '
        for key in clf.best_params_ : msg += '{} -> {} ; '.format(key, clf.best_params_[key])
        print('  ~ Best estimator build ...')
        print(msg)
        print('  ~ Best score of {} ...'.format(clf.best_score_))
        # Save the best estimator as attribute
        self.model = clf.best_estimator_
        del X_tr, y_tr, prm, clf, msg

    # Launch the random searched RandomForest model
    def random_forest(self, n_iter=50, verbose=0) :

        # Prepares the data
        X_tr, y_tr = shuffle(np.hstack((self.f_t, self.h_t)), self.l_t)
        # Defines the model
        clf = RandomForestClassifier(bootstrap=True, n_jobs=self.njobs, criterion='entropy')
        prm = {'n_estimators': randint(150, 250), 'max_depth': randint(10, 30), 'max_features': ['sqrt', None]}
        # Launching the fine-tuning
        clf = RandomizedSearchCV(clf, verbose=verbose, cv=5, param_distributions=prm, n_iter=n_iter, 
                                 scoring=['accuracy', 'neg_log_loss', 'f1_weighted'], refit='accuracy')
        # Log
        print('|-> Learning on {} vectors through cross-validation ...'.format(X_tr.shape[0]))
        # Launch the model
        clf.fit(X_tr, y_tr, sample_weight=sample_weight(y_tr))
        # Log results
        msg = '  ~ '
        for key in clf.best_params_ : msg += '{} -> {} ; '.format(key, clf.best_params_[key])
        print('  ~ Best estimator build ...')
        print(msg)
        print('  ~ Best score of {} ...'.format(clf.best_score_))
        # Save the best estimator as attribute
        self.model = clf.best_estimator_
        del X_tr, y_tr, clf, prm, msg

    # Launch the multi-channels 1D-convolution model
    def conv_1D(self, max_epochs=100, verbose=0) :

        # Truncate the learning to a maximum of cpus
        from keras import backend as K
        K.set_image_dim_ordering('th')
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Prepares the data
        X_tr, y_tr = shuffle(self.r_t, self.l_t)
        X_tr = reformat_vectors(X_tr, self.name, reduced=self.reduced, red_index=self.red_idx)

        # Build model
        def build_model(inputs, output_size) :

            def conv_input(inp) :

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
                mod = GlobalAveragePooling1D()(mod)

                return mod

            mod = merge([conv_input(inp) for inp in inputs])
            mod = Dense(200)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.25)(mod)
            mod = Dense(100)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.25)(mod)
            mod = Dense(100)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = GaussianDropout(0.25)(mod)
            mod = Dense(output_size, activation='softmax')(mod)
            
            return mod

        # Define the inputs
        inp = [Input(shape=X_tr[0][0].shape) for num in range(len(X_tr))]
        # Build model
        model = Model(inputs=inp, outputs=[build_model(inp, len(np.unique(y_tr)))])
        # Compile and launch the model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        model.fit(X_tr, np_utils.to_categorical(y_tr), batch_size=32, epochs=max_epochs, 
                  verbose=verbose, validation_split=0.2, shuffle=True, callbacks=[early])
        # Save as attribute
        self.model = model
        # Memory efficiency
        del X_tr, y_tr, inp, early, model

    # Previous model enhanced with features in neural network
    def deep_conv_1D(self, size_merge=100, max_epochs=100, verbose=0) :

        # Truncate the learning to a maximum of cpus
        from keras import backend as K
        K.set_image_dim_ordering('tf')
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Prepares the data
        X_tr, f_tr, h_tr, y_tr = shuffle(self.r_t, self.f_t, self.h_t, self.l_t)
        X_tr = reformat_vectors(X_tr, self.name, reduced=self.reduced, red_index=self.red_idx)
        # Build inputs for convolution
        inputs = [Input(shape=X_tr[0][0].shape) for num in range(len(X_tr))]

        # Build convolution model
        def conv_input(inp, size_merge) :

            mod = Conv1D(100, 64)(inp)
            mod = BatchNormalization(axis=1, momentum=0.9, center=True, scale=True)(mod)
            mod = Activation('relu')(mod)
            mod = MaxPooling1D(pool_size=2)(mod)
            mod = Dropout(0.1)(mod)
            mod = Conv1D(100, 32)(mod)
            mod = BatchNormalization(axis=1, momentum=0.9, center=True, scale=True)(mod)
            mod = Activation('relu')(mod)
            mod = GaussianDropout(0.25)(mod)
            mod = GlobalAveragePooling1D()(mod)
            mod = Dense(size_merge, activation='relu')(mod)

            return mod

        inp1 = Input(shape=(f_tr.shape[1],))
        mod1 = Dense(300)(inp1)
        mod1 = BatchNormalization()(mod1)
        mod1 = Activation('tanh')(mod1)
        mod1 = Dropout(0.25)(mod1)
        mod1 = Dense(size_merge, activation='relu')(mod1)

        inp2 = Input(shape=(h_tr.shape[1],))
        mod2 = Dense(300)(inp2)
        mod2 = BatchNormalization()(mod2)
        mod2 = Activation('tanh')(mod2)
        mod2 = Dropout(0.25)(mod2)
        mod2 = Dense(size_merge, activation='relu')(mod2)

        mod = merge([conv_input(inp, size_merge) for inp in inputs] + [mod1, mod2])
        mod = BatchNormalization()(mod)
        mod = Dense(500)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.2)(mod)
        mod = Dense(500)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.2)(mod)
        mod = Dense(250)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = GaussianDropout(0.2)(mod)
        mod = Dense(len(np.unique(y_tr)), activation='softmax')(mod)

        # Final build of model
        model = Model(inputs=inputs+[inp1, inp2], outputs=mod)
        # Compile and launch the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        model.fit(X_tr + [f_tr, h_tr], np_utils.to_categorical(y_tr), batch_size=32, epochs=max_epochs, 
                  verbose=verbose, validation_split=0.2, shuffle=True, callbacks=[early])
        # Save as attribute
        self.model = model
        # Memory efficiency
        del X_tr, y_tr, f_tr, inp, early, model

    # Launch the one-channeled 2D-convolution model
    def conv_2D(self, max_epochs=100, verbose=0) :

        # Truncate the learning to a maximum of cpus
        from keras import backend as K
        K.set_image_dim_ordering('th')
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Prepares the data
        X_tr, y_tr = shuffle(self.r_t, self.l_t)
        X_tr = reformat_vectors(X_tr, self.name, reduced=self.reduced, red_index=self.red_idx)
        # Build model
        model = Sequential()
        # Convolutionnal layers
        model.add(Convolution2D(64, (8, 80), input_shape=X_tr[0].shape, data_format='channels_first'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=1, momentum=0.95))
        model.add(MaxPooling2D(pool_size=(1, 1.5), data_format='channels_first'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(128, (1, 30), data_format='channels_first'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=1, momentum=0.925))
        model.add(MaxPooling2D(pool_size=(1, 1.5), data_format='channels_first'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(256, (1, 10), data_format='channels_first'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=1, momentum=0.9))
        model.add(GaussianDropout(0.25))
        # Dense network
        model.add(GlobalAveragePooling2D(data_format='channels_first'))
        model.add(Dense(500))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(100))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(len(np.unique(y_tr))))
        model.add(Activation('softmax'))
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        # Create the callbacks
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        model.fit(X_tr, np_utils.to_categorical(y_tr), batch_size=32, epochs=max_epochs, 
                  verbose=verbose, validation_split=0.2, shuffle=True, callbacks=[early])
        # Save as attribute
        self.model = model
        # Memory efficiency
        del X_tr, y_tr, early, model

    # Previous model enhanced with features in neural network
    def deep_conv_2D(self, size_merge=100, max_epochs=100, verbose=0) :

        # Truncate the learning to a maximum of cpus
        from keras import backend as K
        K.set_image_dim_ordering('th')
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Prepares the data
        X_tr, f_tr, h_tr, y_tr = shuffle(self.r_t, self.f_t, self.h_t, self.l_t)
        X_tr = reformat_vectors(X_tr, self.name, reduced=self.reduced, red_index=self.red_idx)
        # Build inputs for convolution
        inp0 = Input(shape=X_tr[0].shape)
        mod0 = Convolution2D(64, (8, 60), data_format='channels_first')(inp0)
        mod0 = Activation('relu')(mod0)
        mod0 = BatchNormalization(axis=1)(mod0)
        mod0 = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod0)
        mod0 = Dropout(0.25)(mod0)
        mod0 = Convolution2D(128, (1, 30), data_format='channels_first')(mod0)
        mod0 = Activation('relu')(mod0)
        mod0 = BatchNormalization(axis=1)(mod0)
        mod0 = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod0)
        mod0 = Dropout(0.25)(mod0)
        mod0 = Flatten()(mod0)
        mod0 = Dense(100)(mod0)
        mod0 = BatchNormalization()(mod0)
        mod0 = Activation('tanh')(mod0)
        mod0 = Dropout(0.25)(mod0)
        mod0 = Dense(size_merge, activation='tanh')(mod0)
        # Input features
        inp1 = Input(shape=(f_tr.shape[1],))
        mod1 = Dense(250)(inp1)
        mod1 = BatchNormalization()(mod1)
        mod1 = Activation('tanh')(mod1)
        mod1 = Dropout(0.25)(mod1)
        mod1 = Dense(150)(mod1)
        mod1 = BatchNormalization()(mod1)
        mod1 = Activation('tanh')(mod1)
        mod1 = Dropout(0.25)(mod1)
        mod1 = Dense(size_merge, activation='tanh')(mod1)
        # Handcrafted features
        inp2 = Input(shape=(h_tr.shape[1],))
        mod2 = Dense(250)(inp2)
        mod2 = BatchNormalization()(mod2)
        mod2 = Activation('tanh')(mod2)
        mod2 = Dropout(0.25)(mod2)
        mod2 = Dense(150)(mod2)
        mod2 = BatchNormalization()(mod2)
        mod2 = Activation('tanh')(mod2)
        mod2 = Dropout(0.25)(mod2)
        mod2 = Dense(size_merge, activation='tanh')(mod2)
        # Merge both channels
        mod = merge([mod0, mod1, mod2])
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.25)(mod)
        mod = Dense(150)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.25)(mod)
        mod = Dense(150)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = GaussianDropout(0.25)(mod)
        mod = Dense(75)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.25)(mod)
        mod = Dense(len(np.unique(y_tr)), activation='softmax')(mod)
        # Final build of model
        model = Model(inputs=[inp0, inp1, inp2], outputs=mod)
        # Compile and launch the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        model.fit([X_tr, f_tr, h_tr], np_utils.to_categorical(y_tr), batch_size=32, epochs=max_epochs, 
                  verbose=verbose, validation_split=0.2, shuffle=True, callbacks=[early])
        # Save as attribute
        self.model = model
        # Memory efficiency
        del X_tr, y_tr, f_tr, inp1, inp0, mod, mod0, early, model

    # Defines a multi-channel LSTM
    def LSTM(self, size_merge=100, max_epochs=100, verbose=0) :

        # Truncate the learning to a maximum of cpus
        from keras import backend as K
        K.set_image_dim_ordering('th')
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Prepares the data
        X_tr, y_tr = shuffle(self.r_t, self.l_t)
        X_tr = reformat_vectors(X_tr, self.name, reduced=self.reduced, red_index=self.red_idx)
        print(X_tr[0].shape)
        # Build model
        def build_model(inputs, output_size) :

            def LSTM_input(inp) :

                mod = LSTM(500)(inp)
                mod = BatchNormalization()(mod)
                mod = Activation('tanh')(mod)
                mod = Dropout(0.30)(mod)
                mod = LSTM(500)(mod)
                mod = BatchNormalization()(mod)
                mod = Activation('tanh')(mod)
                mod = Dropout(0.30)(mod)
                mod = Dense(size_merge, activation='relu')(mod)

                return mod

            mod = merge([LSTM_input(inp) for inp in inputs])
            mod = Dense(200)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.25)(mod)
            mod = Dense(100)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.25)(mod)
            mod = Dense(100)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = GaussianDropout(0.25)(mod)
            mod = Dense(output_size, activation='softmax')(mod)
            
            return mod

        # Define the inputs
        inp = [Input(shape=(None, X_tr[0].shape[1], X_tr[0].shape[2])) for num in range(len(X_tr))]
        # Build model
        model = Model(inputs=inp, outputs=[build_model(inp, len(np.unique(y_tr)))])
        # Compile and launch the model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        model.fit(X_tr, np_utils.to_categorical(y_tr), batch_size=32, epochs=max_epochs, 
                  verbose=verbose, validation_split=0.2, shuffle=True, callbacks=[early])
        # Save as attribute
        self.model = model
        # Memory efficiency
        del X_tr, y_tr, inp, early, model

    # Defines an enhanced LSTM classficiation model
    def deep_LSTM(self, size_merge=100, max_epochs=100, verbose=0) :

        # Truncate the learning to a maximum of cpus
        from keras import backend as K
        K.set_image_dim_ordering('th')
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Prepares the data
        X_tr, y_tr = shuffle(self.r_t, self.l_t)
        X_tr = reformat_vectors(X_tr, self.name, reduced=self.reduced, red_index=self.red_idx)
        # Build inputs for convolution
        inputs = [Input(shape=X_tr[0][0].shape) for num in range(len(X_tr))]

        def LSTM_input(inp, size_merge) :

            mod = Bidirectionnal(LSTM(500, return_sequences=True))(inp)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.30)(mod)
            mod = Bidirectionnal(LSTM(250, return_sequences=False))(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.30)(mod)
            mod = Dense(size_merge, activation='relu')(mod)

            return mod

        inp1 = Input(shape=(f_tr.shape[1],))
        mod1 = Dense(300)(inp1)
        mod1 = BatchNormalization()(mod1)
        mod1 = Activation('tanh')(mod1)
        mod1 = Dropout(0.25)(mod1)
        mod1 = Dense(size_merge, activation='relu')(mod1)

        inp2 = Input(shape=(h_tr.shape[1],))
        mod2 = Dense(300)(inp2)
        mod2 = BatchNormalization()(mod2)
        mod2 = Activation('tanh')(mod2)
        mod2 = Dropout(0.25)(mod2)
        mod2 = Dense(size_merge, activation='relu')(mod2)

        mod = merge([conv_input(inp) for inp in inputs])
        mod = Dense(200)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.25)(mod)
        mod = Dense(100)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.25)(mod)
        mod = Dense(100)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = GaussianDropout(0.25)(mod)
        mod = Dense(output_size, activation='softmax')(mod)

        # Final build of model
        model = Model(inputs=inputs+[inp1, inp2], outputs=mod)
        # Compile and launch the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        model.fit(X_tr + [f_tr, h_tr], np_utils.to_categorical(y_tr), batch_size=32, epochs=max_epochs, 
                  verbose=verbose, validation_split=0.2, shuffle=True, callbacks=[early])
        # Save as attribute
        self.model = model
        # Memory efficiency
        del X_tr, y_tr, f_tr, inp, early, model

    # Estimates the performance of the model
    def performance(self) :

        # Compute the probabilities
        if self.name in self.case_raw :
            pbs = self.model.predict(reformat_vectors(self.r_e, self.name, reduced=self.reduced, red_index=self.red_idx))
            dtf = score_verbose(self.l_e, [np.argmax(ele) for ele in pbs])
            del pbs
        elif self.name in self.case_bth :
            X_va = reformat_vectors(self.r_e, self.name, reduced=self.reduced, red_index=self.red_idx)
            if self.name == 'DeepConv1D' : X_va = X_va + [self.f_e, self.h_e]
            elif self.name == 'DeepConv2D' : X_va = [X_va, self.f_e, self.h_e]
            pbs = self.model.predict(X_va)
            dtf = score_verbose(self.l_e, [np.argmax(ele) for ele in pbs])
            del X_va, pbs
        elif self.name in self.case_fea : 
            pbs = self.model.predict_proba(np.hstack((self.f_e, self.h_e)))
            dtf = score_verbose(self.l_e, np.asarray([np.argmax(ele) for ele in pbs]))
            del pbs
        # Return results
        return dtf

    # Display the importances when the trees are trained
    def plot_importances(self, n_features=20) :

        imp = self.model.feature_importances_
        idx = np.argsort(imp)[::-1]
        imp = imp[idx][:n_features]
        with open('./Fea_Data/features.txt') as raw : lab = raw.readlines()
        plt.figure(figsize=(18,10))
        plt.title('Feature Importances - {}'.format(self.name))
        plt.barh(range(len(imp)), imp, color="lightblue", align="center")
        plt.yticks(range(len(imp)), lab[idx][:len(imp)])
        plt.ylim([-1, len(imp)])
        plt.show()

    # To launch it from everywhere
    def save_model(self) :

        if self.name in self.case_fea : 
            if self.truncate : joblib.dump(self.model, './Truncates/clf_{}.h5'.format(self.name))
            else : joblib.dump(self.model, './Classifiers/clf_{}.h5'.format(self.name))
        elif self.name in self.case_raw + self.case_bth :
            if self.truncate : self.model.save('./Truncates/clf_{}.h5'.format(self.name))
            else : self.model.save('./Classifiers/clf_{}.h5'.format(self.name))

    # Lazy function if necessary
    def load_model(self) :

        if self.name in self.case_fea : 
            self.model = joblib.load('./Classifiers/clf_{}.h5'.format(self.name))
        elif self.name in self.case_raw + self.case_bth : 
            self.model = load_model('./Classifiers/clf_{}.h5'.format(self.name))

        # Return the model
        return self

    # Defines a launcher
    def learn(self, n_iter=50, max_epochs=100, verbose=0) :

        # Launch the learning process
        if self.name == 'XGBoost' : self.xgboost(n_iter=n_iter, verbose=verbose)
        elif self.name == 'RandomForest' : self.random_forest(n_iter=n_iter, verbose=verbose)
        elif self.name == 'Conv1D' : self.conv_1D(max_epochs=max_epochs, verbose=verbose)
        elif self.name == 'Conv2D' :  self.conv_2D(max_epochs=max_epochs, verbose=verbose)
        elif self.name == 'DeepConv1D' : self.deep_conv_1D(max_epochs=max_epochs, verbose=verbose)
        elif self.name == 'DeepConv2D' : self.deep_conv_2D(max_epochs=max_epochs, verbose=verbose)
        # Return object
        return self
