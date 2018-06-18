# Author : DINDIN Meryll
# Date : 16/11/2017

from package.toolbox import *
from hyperband.optimizer import *

# Defines a structure for the machine-learning models

class ML_Model:

    # Initialization
    # path refers to the absolute path towards the datasets
    # threads refers to the amount of affordable threads
    def __init__(self, path, threads=multiprocessing.cpu_count()):

        # Attributes
        self.inp = path
        self.njobs = threads

        # Apply on the data
        with h5py.File(self.inp, 'r') as dtb:
            # Load the labels and initialize training and testing sets
            self.l_t = dtb['label_t'].value           
            self.l_e = dtb['label_e'].value
            try: self.l_v = dtb['label_v'].value
            except: pass
            # Memory efficiency
            del msk_labels

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

        # Defines the different folds on which to apply the Hyperband
        self.folds = KFold(n_splits=5, shuffle=True)

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
        res = sorted(res, key = lambda x: x[key])[0]

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
        if marker: out = './models/{}.pk'.format(nme)
        else: out = './models/{}_{}.pk'.format(nme, marker)
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
            kap = kappa_score(true, prd)
            tle = '{} | {} | Accuracy: {:.2%} | Kappa: {:.2%}'
            plt.title(tle.format(title, pth, acc, kap))
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

