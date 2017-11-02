# Author : DINDIN Meryll
# Date : 01/11/2017

from loading import *

# Build the models as initialized
class Models :

    # Initialization
    def __init__(self, model, max_jobs=multiprocessing.cpu_count()-1, reduced=False, red_index=[6, 7]) :

        self.model = model
        self.njobs = max_jobs
        # Differentiate cases
        self.case_fea = ['XGBoost', 'RandomForest']
        self.case_raw = ['Conv1D']
        self.case_bth = ['DeepConv']
        # Default arguments for convolution
        self.reduced = reduced
        self.red_idx = red_index
        # Load the data according to the model
        if model in self.case_fea : self.loader = Loader().load_fea()
        elif model in self.case_raw : self.loader = Loader().load_raw()
        elif model in self.case_bth : self.loader = Loader().load_bth()

    # Launch the random searched XGBoost model
    def xgboost(self, n_iter=50, verbose=0) :

        # Prepares the data
        X_tr, y_tr = shuffle(remove_columns(self.loader.train, ['Subject', 'Labels']).values, self.loader.train['Labels'].values.ravel().astype(int) - 1)
        # Defines the model
        clf = xgboost.XGBClassifier(nthread=self.njobs)
        prm = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0], 'max_depth': randint(10, 30),
               'n_estimators': randint(250, 350),'gamma': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
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
        X_tr, y_tr = shuffle(remove_columns(self.loader.train, ['Subject', 'Labels']).values, self.loader.train['Labels'].values.ravel().astype(int) - 1)
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
    def conv1D(self, max_epochs=100, verbose=0) :

        # Truncate the learning to a maximum of cpus
        from keras import backend as K
        K.set_image_dim_ordering('th')
        S = tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=self.njobs))
        K.set_session(S)
        # Prepares the data
        X_tr, y_tr = shuffle(self.loader.X_tr, self.loader.y_tr)
        X_tr = reformat_vectors(X_tr, reduced=self.reduced, red_index=self.red_idx)

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
                mod = GaussianDropout(0.75)(mod)

                return mod

            mod = merge([conv_input(inp) for inp in inputs])
            mod = Flatten()(mod)
            mod = Dense(400)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = Dropout(0.5)(mod)
            mod = Dense(200)(mod)
            mod = BatchNormalization()(mod)
            mod = Activation('tanh')(mod)
            mod = GaussianDropout(0.75)(mod)
            mod = Dense(output_size, activation='softmax')(mod)
            return mod

        # Define the inputs
        inp = [Input(shape=X_tr[0][0].shape) for num in range(len(X_tr))]
        # Build model
        model = Model(inputs=inp, outputs=[build_model(inp, len(np.unique(y_tr)))])
        # Compile and launch the model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=7, verbose=0, mode='auto')
        for ele in list(np.unique(y_tr)) :
            print('  ~ Class {} with Ratio Of {} ...'.format(int(ele), round(float(len(np.where(y_tr == ele)[0])) / float(len(y_tr)), 2)))
        model.fit(X_tr, np_utils.to_categorical(y_tr), batch_size=32, epochs=max_epochs, verbose=verbose, validation_split=0.2, shuffle=True)
        # Save as attribute
        self.model = model
        # Memory efficiency
        del X_tr, y_tr, inp, early, model

    # Estimates the performance of the model
    def performance(self) :

        # Compute the probabilities
        if self.model in self.case_raw : pbs = self.model.predict(reformat_vectors(self.loader.X_va, reduced=self.reduced, red_index=self.red_idx))
        elif self.model in self.case_bth : pass
        else : pbs = self.model.predict_proba(self.loader.X_va)
        # Performance on the testing subset 
        print('\n|-> Main scores on test subset :')
        score_verbose(self.loader.y_va, [np.argmax(ele) for ele in pbs])

    # Defines a launcher
    def learn(self, n_iter=50, max_epochs=100, verbose=0) :

        # Launch the learning process
        if self.model == 'XGBoost' : self.xgboost(n_iter=n_iter, verbose=verbose)
        elif self.model == 'RandomForest' : self.rarndom_forest(n_iter=n_iter, verbose=verbose)
        elif self.model == 'Conv1D' : self.conv1D(max_epochs=max_epochs, verbose=verbose)
        # Print the performance
        self.performance()