# Author : DINDIN Meryll
# Date : 01/11/2017

from loading import *

# Build the models as initialized
class Models :

    # Initialization
    def __init__(self, model, init=False, max_jobs=multiprocessing.cpu_count()-1, reduced=False, red_index=[6, 7]) :

        self.name = model
        self.njobs = max_jobs
        # Differentiate cases
        self.case_fea = ['XGBoost', 'RandomForest']
        self.case_raw = ['Conv1D', 'Conv2D']
        self.case_bth = ['DeepConv1D', 'DeepConv2D']
        # Default arguments for convolution
        self.reduced = reduced
        self.red_idx = red_index
        # Load the data according to the model
        if model in self.case_fea :
            if init : self.loader = Loader().load_fea()
            else : 
                with open('./Loaders/loader_fea.pickle', 'rb') as raw : self.loader = pickle.load(raw)
        elif model in self.case_raw : 
            if init : self.loader = Loader().load_raw()
            else : 
                with open('./Loaders/loader_raw.pickle', 'rb') as raw : self.loader = pickle.load(raw)
        elif model in self.case_bth : 
            if init : self.loader = Loader().load_bth()
            else : 
                with open('./Loaders/loader_bth.pickle', 'rb') as raw : self.loader = pickle.load(raw)

    # Launch the random searched XGBoost model
    def xgboost(self, n_iter=50, verbose=0) :

        # Prepares the data
        X_tr, y_tr = shuffle(remove_columns(self.loader.train, ['Subjects', 'Labels']).values, self.loader.train['Labels'].values.ravel().astype(int) - 1)
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

    # Launch the random searched RandomForest model
    def random_forest(self, n_iter=50, verbose=0) :

        # Prepares the data
        X_tr, y_tr = shuffle(remove_columns(self.loader.train, ['Subjects', 'Labels']).values, self.loader.train['Labels'].values.ravel().astype(int) - 1)
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

    # Launch the multi-channels 1D-convolution model
    def conv_1D(self, max_epochs=100, verbose=0) :

        # Truncate the learning to a maximum of cpus
        from keras import backend as K
        K.set_image_dim_ordering('th')
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Prepares the data
        X_tr, y_tr = shuffle(self.loader.X_tr, self.loader.y_tr)
        X_tr = reformat_vectors(X_tr, self.name, reduced=self.reduced, red_index=self.red_idx)

        # Build model
        def build_model(inputs, output_size) :

            def conv_input(inp) :

                mod = Conv1D(200, 50)(inp)
                mod = BatchNormalization(axis=1, momentum=0.9, center=True, scale=True)(mod)
                mod = Activation('relu')(mod)
                mod = MaxPooling1D(pool_size=2)(mod)
                mod = Dropout(0.5)(mod)
                mod = Conv1D(50, 25)(mod)
                mod = BatchNormalization(axis=1, momentum=0.9, center=True, scale=True)(mod)
                mod = Activation('relu')(mod)
                mod = GaussianDropout(0.5)(mod)

                return mod

            mod = merge([conv_input(inp) for inp in inputs])
            mod = Flatten()(mod)
            mod = Dense(500)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.5)(mod)
            mod = Dense(100)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = GaussianDropout(0.5)(mod)
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

    # Launch the one-channeled 2D-convolution model
    def conv_2D(self, max_epochs=100, verbose=0) :

        # Truncate the learning to a maximum of cpus
        from keras import backend as K
        K.set_image_dim_ordering('th')
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Prepares the data
        X_tr, y_tr = shuffle(self.loader.X_tr, self.loader.y_tr)
        X_tr = reformat_vectors(X_tr, self.name, reduced=self.reduced, red_index=self.red_idx)
        # Build model
        model = Sequential()
        # Convolutionnal layers
        model.add(Convolution2D(64, (8, 50), input_shape=X_tr[0].shape, data_format='channels_first'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(1, 2), data_format='channels_first'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(64, (1, 25), data_format='channels_first'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(1, 5), data_format='channels_first'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(64, (1, 25), data_format='channels_first'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=1, momentum=0.9, center=True, scale=True))
        model.add(GaussianDropout(0.5))
        # Dense network
        model.add(GlobalAveragePooling2D(data_format='channels_first'))
        model.add(Dense(1000))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(500))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
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
    def deep_conv_1D(self, size_merge=100, max_epochs=100, verbose=0) :

        # Truncate the learning to a maximum of cpus
        from keras import backend as K
        K.set_image_dim_ordering('th')
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Prepares the data
        X_tr, y_tr = shuffle(self.loader.X_tr, self.loader.y_tr)
        X_tr = reformat_vectors(X_tr, self.name, reduced=self.reduced, red_index=self.red_idx)
        # Build inputs for convolution
        inputs = [Input(shape=X_tr[0][0].shape) for num in range(len(X_tr))]

        # Build convolution model
        def conv_input(inp, size_merge) :

            mod = Conv1D(200, 50)(inp)
            mod = BatchNormalization(axis=1, momentum=0.9, center=True, scale=True)(mod)
            mod = Activation('relu')(mod)
            mod = MaxPooling1D(pool_size=2)(mod)
            mod = Dropout(0.5)(mod)
            mod = Conv1D(50, 25)(mod)
            mod = BatchNormalization(axis=1, momentum=0.9, center=True, scale=True)(mod)
            mod = Activation('relu')(mod)
            mod = GaussianDropout(05)(mod)
            mod = GlobalAveragePooling1D()(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.5)(mod)
            mod = Dense(size_merge, activation='tanh')(mod)

            return mod

        inp1 = Input(shape=(self.loader.train.shape[1],))
        mod1 = Dense(200)(inp1)
        mod1 = BatchNormalization()(mod1)
        mod1 = Activation('tanh')(mod1)
        mod1 = GaussianDropout(0.5)(mod1)
        mod1 = Dense(size_merge, activation='tanh')(mod1)

        mod = merge([conv_input(inp, size_merge) for inp in inputs] + [mod1])
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.5)(mod)
        mod = Dense(500)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = Dropout(0.5)(mod)
        mod = Dense(100)(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('tanh')(mod)
        mod = GaussianDropout(0.75)(mod)
        mod = Dense(len(np.unique(y_tr)), activation='softmax')(mod)

        # Final build of model
        model = Model(inputs=inputs+[inp1], outputs=mod)
        # Compile and launch the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=20, verbose=0, mode='auto')
        model.fit(X_tr + [self.loader.train.values], np_utils.to_categorical(y_tr), batch_size=32, epochs=max_epochs, 
                  verbose=verbose, validation_split=0.2, shuffle=True, callbacks=[early])
        # Save as attribute
        self.model = model
        # Memory efficiency
        del X_tr, y_tr, inp, early, model

    # Estimates the performance of the model
    def performance(self) :

        # Compute the probabilities
        if self.name in self.case_raw :
            pbs = self.model.predict(reformat_vectors(self.loader.X_va, self.name, reduced=self.reduced, red_index=self.red_idx))
            dtf = score_verbose(self.loader.y_va, [np.argmax(ele) for ele in pbs])
            del pbs
        elif self.name in self.case_bth :
            X_va = reformat_vectors(self.loader.X_va, self.name, reduced=self.reduced, red_index=self.red_idx)
            X_va = X_va + [self.loader.valid.values]
            pbs = self.model.predict(X_va)
            dtf = score_verbose(self.loader.y_va, [np.argmax(ele) for ele in pbs])
            del X_va, pbs
        elif self.name in self.case_fea : 
            pbs = self.model.predict_proba(remove_columns(self.loader.valid, ['Subjects', 'Labels']).values)
            dtf = score_verbose(self.loader.valid['Labels'].values.ravel(), [np.argmax(ele) for ele in pbs])
            del pbs
        # Return results
        return dtf

    # Display the importances when the trees are trained
    def plot_importances(self, n_features=20) :

        imp = self.model.feature_importances_
        idx = np.argsort(imp)[::-1]
        imp = imp[idx][:n_features]
        plt.figure(figsize=(18,10))
        plt.title('Feature Importances - {}'.format(self.name))
        plt.barh(range(len(imp)), imp, color="lightblue", align="center")
        plt.yticks(range(len(imp)), self.loader.train.columns[idx][:len(imp)])
        plt.ylim([-1, len(imp)])
        plt.show()

    # To launch it from everywhere
    def save_model(self) :

        if self.name in self.case_fea : 
            joblib.dump(self.model, './Classifiers/clf_{}.h5'.format(self.name))
        elif self.name in self.case_raw + self.case_bth : 
            self.model.save('./Classifiers/clf_{}.h5'.format(self.name))

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
        # Return object
        return self
