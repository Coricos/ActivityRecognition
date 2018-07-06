# Author : DINDIN Meryll
# Date : 16/11/2017

from package.toolbox import *
from hyperband.optimizer import *

# Defines a structure for the machine-learning models

class ML_Model:

    # Initialization
    # path refers to the absolute path towards the datasets
    # threads refers to the amount of affordable threads
    def __init__(self, path=None, threads=multiprocessing.cpu_count()):

        # Attributes
        self.threads = threads

        if path:
            self.inp = path
            # Apply on the data
            with h5py.File(self.inp, 'r') as dtb:
                # Load the labels and initialize training and testing sets
                self.l_t = dtb['label_t'].value           
                self.l_e = dtb['label_e'].value
                try: self.l_v = dtb['label_v'].value
                except: pass

            # Suppress the missing labels for categorical learning
            self.lbe = LabelEncoder()
            tmp = np.unique(list(self.l_t) + list(self.l_e))
            self.lbe.fit(tmp)
            self.l_t = self.lbe.transform(self.l_t)
            self.l_e = self.lbe.transform(self.l_e)
            try: self.l_v = self.lbe.transform(self.l_v)
            except: pass
            # Define the specific anomaly issue
            self.num_classes = len(tmp)
            # Memory efficiency
            del tmp

            # Load the data
            with h5py.File(self.inp, 'r') as dtb:
                self.train = np.hstack((dtb['fea_t'].value, dtb['fft_a_t'].value, dtb['fft_g_t'].value))
                self.valid = np.hstack((dtb['fea_e'].value, dtb['fft_a_e'].value, dtb['fft_g_e'].value))
                try: self.evals = np.hstack((dtb['fea_v'].value, dtb['fft_a_v'].value, dtb['fft_g_v'].value))
                except: pass

    # Application of the ML models
    # nme refers to the type of model to use
    # marker allows specific learning instance
    # max_iter refers to the amount of iterations with the hyperband algorithm
    def learn(self, nme, marker=None, max_iter=100):

        # Defines the data representation folder
        val = dict()
        val['x_train'] = self.train
        val['y_train'] = self.l_t
        val['w_train'] = sample_weight(self.l_t)
        val['x_valid'] = self.valid
        val['y_valid'] = self.l_e
        val['w_valid'] = sample_weight(self.l_e)

        # Defines the random search through cross-validation
        hyp = Hyperband(get_params, try_params, max_iter=max_iter, n_jobs=self.threads)
        res = hyp.run(nme, val, skip_last=1)
        res = sorted(res, key = lambda x: x['acc'])[0]

        # Extract the best estimator
        if nme == 'RFS':
            mod = RandomForestClassifier(**res['params'])
        if nme == 'GBT':
            mod = GradientBoostingClassifier(**res['params'])
        if nme == 'LGB':
            mod = lgb.LGBMClassifier(objective='multiclass', **res['params'])
        if nme == 'ETS':
            mod = ExtraTreesClassifier(**res['params'])
        if nme == 'XGB':
            mod = xgb.XGBClassifier(**res['params'])
        if nme == 'SGD':
            mod = SGDClassifier(**res['params'])
        # Refit the best model
        mod.fit(val['x_train'], val['y_train'], sample_weight=val['w_train'])

        # Serialize the best obtained model
        if marker: out = './models/{}_{}.pk'.format(nme, marker)
        else: out = './models/{}.pk'.format(nme)
        joblib.dump(mod, out)

    # Defines the confusion matrix on train, test and validation sets
    # nme refers to a new path if necessary
    # marker allows specific redirection
    # on_test and on_validation refers to booleans for confusion matrixes
    def confusion_matrix(self, nme, marker=None, on_test=False, on_validation=True):

        # Avoid unnecessary logs
        warnings.simplefilter('ignore')

        # Load the model if necessary
        if marker is None: self.mod = './models/{}.pk'.format(nme)
        else: self.mod = './models/{}_{}.pk'.format(nme, marker)
        clf = joblib.load(self.mod)

        # Method to build and display the confusion matrix
        def build_matrix(prd, true, title):

            lab = np.unique(list(prd) + list(true))
            cfm = confusion_matrix(true, prd)
            cfm = pd.DataFrame(cfm, index=lab, columns=lab)

            fig = plt.figure(figsize=(18,6))
            htp = sns.heatmap(cfm, annot=True, fmt='d', linewidths=1.)
            pth = self.mod.split('/')[-1]
            acc = accuracy_score(true, prd)
            tle = '{} | {} | Accuracy: {:.2%}'
            plt.title(tle.format(title, pth, acc))
            htp.yaxis.set_ticklabels(htp.yaxis.get_ticklabels(), 
                rotation=0, ha='right', fontsize=12)
            htp.xaxis.set_ticklabels(htp.xaxis.get_ticklabels(), 
                rotation=45, ha='right', fontsize=12)
            plt.ylabel('True label') 
            plt.xlabel('Predicted label')
            plt.show()

        if on_test:
            # Compute the predictions for test set
            prd = clf.predict(self.valid)
            build_matrix(prd, self.l_e, 'TEST')
            del prd

        if on_validation:
            # Compute the predictions for validation set
            with h5py.File(self.inp, 'r') as dtb:
                val = np.hstack((dtb['fea_v'].value, dtb['fft_a_v'].value, dtb['fft_g_v'].value))
            prd = clf.predict(val)
            build_matrix(prd, self.l_v, 'VALID')
            del val, prd

    def score(self, nme, marker=None):

        # Avoid unnecessary logs
        warnings.simplefilter('ignore')

        # Load the model if necessary
        if marker: self.mod = './models/{}_{}.pk'.format(nme, marker)
        else: self.mod = './models/{}.pk'.format(nme)
        clf = joblib.load(self.mod)

        prd = clf.predict(self.evals)
        acc = accuracy_score(self.l_v, prd)
        f1s = f1_score(self.l_v, prd, average='weighted')

        return acc, f1s

# Defines a structure for a cross_validation

class CV_ML_Model:

    # Initialization
    # path refers to the absolute path towards the datasets
    # k_fold refers to the number of validation rounds
    # msk_labels refers to the labels to suppress
    # threads refers to the amount of affordable threads
    def __init__(self, path, k_fold=7, msk_labels=[], threads=multiprocessing.cpu_count()):

        # Attributes
        self.input = path
        self.njobs = threads

        # Apply on the data
        with h5py.File(self.input, 'r') as dtb:
            # Load the labels and initialize training and testing sets
            self.lab = dtb['label_t'].value
            msk = get_mask(self.lab, lab_to_del=msk_labels)
            self.lab = self.lab[msk]
            # Define the specific anomaly issue
            self.n_c = len(np.unique(self.lab))
            # Defines the vectors
            self.vec = np.hstack((dtb['fea_t'].value, dtb['fft_a_t'].value, dtb['fft_g_t'].value))
            self.vec = self.vec[msk]
            # Prepares the validation data
            self.l_v = dtb['label_v'].value
            msk = get_mask(self.l_v, lab_to_del=msk_labels)
            self.l_v = self.l_v[msk]
            self.val = np.hstack((dtb['fea_v'].value, dtb['fft_a_v'].value, dtb['fft_g_v'].value))
            self.val = self.val[msk]

        # Defines the cross-validation splits
        self.kfs = StratifiedKFold(n_splits=k_fold, shuffle=True)

        # Apply feature filtering based on variance
        vtf = VarianceThreshold(threshold=0.0)
        self.vec = vtf.fit_transform(self.vec)
        self.val = vtf.transform(self.val)
        joblib.dump(vtf, './models/VTF_Selection.jb')

    # CV Launcher
    # nme refers to the type of model to be launched
    # log_file refers to where to store the intermediate scores
    def launch(self, nme, log_file='./models/CV_SCORING.log'):

        for idx, (i_t, i_e) in enumerate(self.kfs.split(self.lab, self.lab)):

            # Build the corresponding tuned model
            mkr = 'CV_{}'.format(idx)
            mod = ML_Model(threads=self.njobs)
            mod.l_t = self.lab[i_t]
            mod.l_e = self.lab[i_e]
            mod.l_v = self.l_v
            mod.train = self.vec[i_t]
            mod.valid = self.vec[i_e]
            mod.evals = self.val
            # Launch the hyperband optimization
            mod.learn(nme, marker=mkr)
            # Retrieve the scores
            a,f = mod.score(nme, marker=mkr)
            # LOG file for those scores
            with open(log_file, 'a') as raw:
                raw.write('# CV_ROUND {} | Accuracy {:3f} | F1Score {:3f} \n'.format(idx, a, f))

            # Memory efficiency
            del mkr, mod, a, f

    # Stacking of predictions
    # nme refers to the name of the estimator
    # scaler refers whether feature extraction has been used
    def confusion_matrix(self, nme, scaler='./models/VTF_Selection.jb'):

        # Initial vector for result storing
        res = np.zeros((len(self.val), self.n_c))

        # Look for all available models
        lst = sorted(glob.glob('./models/{}_CV_*.pk'.format(nme)))
        for mod in lst:
            # Load the model and make the predictions
            mod = joblib.load(mod)
            res += np_utils.to_categorical(mod.predict(self.val), num_classes=self.n_c)

        # Get the most occurent result
        res = np.asarray([np.argmax(ele) for ele in res])
        
        # Method to build and display the confusion matrix
        def build_matrix(prd, true,):

            lab = np.unique(list(prd) + list(true))
            cfm = confusion_matrix(true, prd)
            cfm = pd.DataFrame(cfm, index=lab, columns=lab)

            fig = plt.figure(figsize=(18,6))
            htp = sns.heatmap(cfm, annot=True, fmt='d', linewidths=1.)
            acc = accuracy_score(true, prd)
            f1s = f1_score(true, prd, average='weighted')
            tle = '{} | F1Score: {:.2%} | Accuracy: {:.2%}'
            plt.title(tle.format('VALIDATION', f1s, acc))
            htp.yaxis.set_ticklabels(htp.yaxis.get_ticklabels(), 
                rotation=0, ha='right', fontsize=12)
            htp.xaxis.set_ticklabels(htp.xaxis.get_ticklabels(), 
                rotation=45, ha='right', fontsize=12)
            plt.ylabel('True label') 
            plt.xlabel('Predicted label')
            plt.show()

        build_matrix(res, self.l_v)
