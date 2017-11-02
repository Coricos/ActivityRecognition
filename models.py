# Author : DINDIN Meryll
# Date : 01/11/2017

from loading import *

# Build the models as initialized
class Model :

    # Initialization
    def __init__(self, model, max_jobs=multiprocessing.cpu_count()-1, reduced=False, red_index=[6, 7]) :

        self.njobs = max_jobs
        # Differentiate cases
        self.case_fea = ['XGBoost', 'RandomForest']
        self.case_raw = ['Conv1D']
        self.case_bth = ['DeepConv']
        # Default arguments for convolution
        self.reduced = reduced
        self.red_idx = red_idx
        # Load the data according to the model
        if model in self.case_fea : self.loader = Load().load_fea()
        elif model in self.case_raw : self.loader = Load().load_raw()
        elif model in self.case_bth : self.loader = Load().load_bth()

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
        # Save the best estimator as attribute
        self.model = clf.best_estimator_

    # Launch the multi-channels 1D-convolution model
    def conv1D(self) :

        # Prepares the data
    