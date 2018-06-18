# Author : DINDIN Meryll
# Date : 16/11/2017

from package.toolbox import *
from package.callback import *

# Deep-learning models of mixed channels
class DL_Model :
    
    # Initialization
    # inp_dtb refers to the input database of where to fetch the data
    # channels refers to the channels to use
    # marker refers to the given id of the model
    def __init__(self, inp_dtb, channels, marker=None) :

        # Keep the channels for validation
        self.inp = inp_dtb
        self.cls = channels
        self.inputs, self.models = [], []
        # Labels and their respective masks
        with h5py.File(inp_dtb, 'r') as dtb :
            # Load the labels and initialize training and testing sets
            self.l_t = dtb['label_t'].value
            self.l_e = dtb['label_e'].value
        # Define the number of classes
        self.classes = np.unique(list(self.l_t) + list(self.l_e))
        # Translate the labels for categorical learning
        self.lbe = LabelEncoder()
        self.lbe.fit(self.classes)
        self.l_t = self.lbe.transform(self.l_t)
        self.l_e = self.lbe.transform(self.l_e)
        # Prepares the path for storage
        if marker: 
            self.mod = './models/MOD_{}.weights'.format(marker)
            self.his = './models/HIS_{}.history'.format(marker)
        else: 
            self.mod = './models/MOD.weights'
            self.his = './models/HIS.history'     

    # Defines a generator (training and testing)
    # fmt refers to whether apply it for training or testing
    # batch refers to the batch size
    def train_generator(self, fmt, batch=64):
        
        ind = 0

        while True :
            
            if fmt == 't': ann = self.l_t
            if fmt == 'e': ann = self.l_e
            # Reinitialize when going too far
            if ind + batch >= len(ann) : ind = 0
            # Initialization of data vector
            vec = []

            if self.cls['with_acc_cv1']:

                with h5py.File(self.inp, 'r') as dtb:
                    for idx in ['x', 'y', 'z']: 
                        vec.append(dtb['acc_{}_{}'.format(idx, fmt)][ind:ind+batch])

            if self.cls['with_acc_cv2']:

                with h5py.File(self.inp, 'r') as dtb:
                    shp = dtb['acc_x_{}'.format(fmt)].shape
                    tmp = np.empty((batch, 3, shp[1]))
                    for idx, mkr in zip(range(3), ['x', 'y', 'z']):
                        key = 'acc_{}_{}'.format(mkr, fmt)
                        tmp[:,idx,:] = dtb[key][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, key

            if self.cls['with_n_a_cv1']:

                with h5py.File(self.inp, 'r') as dtb:
                    vec.append(dtb['n_acc_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_gyr_cv1']:

                with h5py.File(self.inp, 'r') as dtb:
                    for idx in ['x', 'y', 'z']: 
                        vec.append(dtb['gyr_{}_{}'.format(idx, fmt)][ind:ind+batch])

            if self.cls['with_gyr_cv2']:

                with h5py.File(self.inp, 'r') as dtb:
                    shp = dtb['acc_x_{}'.format(fmt)].shape
                    tmp = np.empty((batch, 3, shp[1]))
                    for idx, mkr in zip(range(3), ['x', 'y', 'z']):
                        key = 'gyr_{}_{}'.format(mkr, fmt)
                        tmp[:,idx,:] = dtb[key][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, key

            if self.cls['with_n_g_cv1']:

                with h5py.File(self.inp, 'r') as dtb:
                    vec.append(dtb['n_gyr_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_qua_cv2']:

                 with h5py.File(self.inp, 'r') as dtb:
                    shp = dtb['qua_0_{}'.format(fmt)].shape
                    tmp = np.empty((batch, 4, shp[1]))
                    for idx in range(4):
                        key = 'qua_{}_{}'.format(idx, fmt)
                        tmp[:,idx,:] = dtb[key][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, key

            if self.cls['with_fea']:

                with h5py.File(self.inp, 'r') as dtb:
                    vec.append(dtb['fea_{}'.format(fmt)][ind:ind+batch])

            if self.cls['with_fft']:

                with h5py.File(self.inp, 'r') as dtb:
                    vec.append(dtb['fft_a_{}'.format(fmt)][ind:ind+batch])
                    vec.append(dtb['fft_g_{}'.format(fmt)][ind:ind+batch])

            with h5py.File(self.inp, 'r') as dtb:

                # Defines the labels
                lab = dtb['label_{}'.format(fmt)][ind:ind+batch]
                lab = np_utils.to_categorical(lab, num_classes=len(self.classes))
            
            yield(vec, lab)
            # Memory efficiency
            del lab, vec

            ind += batch

    # Defines a generator (testing and validation)
    # fmt refers to whether apply it for testing or validation
    # batch refers to the batch size
    def valid_generator(self, fmt, batch=512):

        if fmt == 'e': 
            sze = len(self.l_e)
        if fmt == 'v': 
            with h5py.File(self.inp, 'r') as dtb: 
                sze = dtb['acc_x_v'].shape[0]

        ind, poi = 0, sze

        while True :
            
            # Reinitialize when going too far
            if ind > sze : ind, poi = 0, sze
            # Initialization of data vector
            vec = []

            if self.cls['with_acc_cv1']:

                with h5py.File(self.inp, 'r') as dtb:
                    for idx in ['x', 'y', 'z']: 
                        vec.append(dtb['acc_{}_{}'.format(idx, fmt)][ind:ind+batch])

            if self.cls['with_acc_cv2']:

                with h5py.File(self.inp, 'r') as dtb:
                    shp = dtb['acc_x_{}'.format(fmt)].shape
                    tmp = np.empty((min(poi, batch), 3, shp[1]))
                    for idx, mkr in zip(range(3), ['x', 'y', 'z']):
                        key = 'acc_{}_{}'.format(mkr, fmt)
                        tmp[:,idx,:] = dtb[key][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, key

            if self.cls['with_n_a_cv1']:

                with h5py.File(self.inp, 'r') as dtb:
                    vec.append(dtb['n_acc_{}'.format(idx, fmt)][ind:ind+batch])

            if self.cls['with_gyr_cv1']:

                with h5py.File(self.inp, 'r') as dtb:
                    for idx in ['x', 'y', 'z']: 
                        vec.append(dtb['gyr_{}_{}'.format(idx, fmt)][ind:ind+batch])

            if self.cls['with_gyr_cv2']:

                with h5py.File(self.inp, 'r') as dtb:
                    shp = dtb['acc_x_{}'.format(fmt)].shape
                    tmp = np.empty((min(poi, batch), 3, shp[1]))
                    for idx, mkr in zip(range(3), ['x', 'y', 'z']):
                        key = 'gyr_{}_{}'.format(mkr, fmt)
                        tmp[:,idx,:] = dtb[key][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, key

            if self.cls['with_n_g_cv1']:

                with h5py.File(self.inp, 'r') as dtb:
                    vec.append(dtb['n_gyr_{}'.format(idx, fmt)][ind:ind+batch])

            if self.cls['with_qua_cv2']:

                 with h5py.File(self.inp, 'r') as dtb:
                    shp = dtb['qua_0_{}'.format(fmt)].shape
                    tmp = np.empty((min(poi, batch), 4, shp[1]))
                    for idx in range(4):
                        key = 'qua_{}_{}'.format(idx, fmt)
                        tmp[:,idx,:] = dtb[key][ind:ind+batch]
                    vec.append(tmp)
                    del shp, tmp, key

            if self.cls['with_fea']:

                with h5py.File(self.inp, 'r') as dtb:
                    vec.append(dtb['fea_{}'.format(idx, fmt)][ind:ind+batch])

            if self.cls['with_fft']:

                with h5py.File(self.inp, 'r') as dtb:
                    vec.append(dtb['fft_a_{}'.format(idx, fmt)][ind:ind+batch])
                    vec.append(dtb['fft_g_{}'.format(idx, fmt)][ind:ind+batch])

            yield(vec)

            ind += batch
            poi -= batch

    # Add 1D convolution model
    # inp refers to a Keras input
    # callback refers to the adaptive dropout callback
    # arg refers the layers arguments
    def build_CONV1D(self, inp, callback, arg) :

        # Build the selected model
        mod = Reshape((inp._keras_shape[1], 1))(inp)
        mod = Conv1D(100, 50, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = MaxPooling1D(pool_size=2)(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(50, 25, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Conv1D(50, 10, **arg)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = GlobalAveragePooling1D()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        self.models.append(mod)
        if inp not in self.inputs: self.inputs.append(inp) 

    # Add 2D convolution model
    # inp refers to a Keras input
    # callback refers to the adaptive dropout callback
    # arg refers the layers arguments
    def build_CONV2D(self, inp, callback, arg) :

        # Build model
        mod = Reshape((1, inp._keras_shape[1], inp._keras_shape[2]))(inp)
        mod = Convolution2D(64, (mod._keras_shape[1], 60), data_format='channels_first', **arg)(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = PReLU()(mod)
        mod = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Convolution2D(128, (1, 30), data_format='channels_first', **arg)(mod)
        mod = BatchNormalization(axis=1)(mod)
        mod = PReLU()(mod)
        mod = GlobalAveragePooling2D()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        self.models.append(mod)
        if inp not in self.inputs: self.inputs.append(inp) 

    # Add dense network for handcrafted features
    # inp refers to a Keras input
    # callback refers to the adaptive dropout callback
    # arg refers the layers arguments
    def build_NDENSE(self, inp, callback, arg) :

        # Build the model
        mod = Dense(inp._keras_shape[1])(inp)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)
        mod = Dense(mod._keras_shape[1] // 2)(mod)
        mod = BatchNormalization()(mod)
        mod = PReLU()(mod)
        mod = AdaptiveDropout(callback.prb, callback)(mod)

        self.models.append(mod)
        if inp not in self.inputs: self.inputs.append(inp) 

    # Build the whole model
    # ini_dropout refers to the initial dropout rate
    # n_tail refers to the amount of layers to deal with concatenated feature map
    def build(self, ini_dropout, n_tail) :

        warnings.simplefilter('ignore')

        # Reinitialize the models
        self.inputs, self.models = [], []

        # Initializers
        self.drp = DecreaseDropout(ini_dropout, 100)
        arg = {'kernel_initializer': 'he_uniform'}

        if self.cls['with_acc_cv1']:
            with h5py.File(self.inp, 'r') as dtb:
                inp = Input(shape=(dtb['acc_x_t'].shape[1],))
            for idx in range(3): self.build_CONV1D(inp, self.drp, arg)

        if self.cls['with_acc_cv2']:
            with h5py.File(self.inp, 'r') as dtb:
                inp = Input(shape=(3, dtb['acc_x_t'].shape[1]))
            self.build_CONV2D(inp, self.drp, arg)

        if self.cls['with_n_a_cv1']:
            with h5py.File(self.inp, 'r') as dtb:
                inp = Input(shape=(dtb['n_acc_t'].shape[1],))
            self.build_CONV1D(inp, self.drp, arg)

        if self.cls['with_gyr_cv1']:
            with h5py.File(self.inp, 'r') as dtb:
                inp = Input(shape=(dtb['gyr_x_t'].shape[1],))
            for idx in range(3): self.build_CONV1D(inp, self.drp, arg)

        if self.cls['with_gyr_cv2']:
            with h5py.File(self.inp, 'r') as dtb:
                inp = Input(shape=(3, dtb['gyr_x_t'].shape[1]))
            self.build_CONV2D(inp, self.drp, arg)

        if self.cls['with_n_g_cv1']:
            with h5py.File(self.inp, 'r') as dtb:
                inp = Input(shape=(dtb['n_gyr_t'].shape[1],))
            self.build_CONV1D(inp, self.drp, arg)

        if self.cls['with_qua_cv2']:
            with h5py.File(self.inp, 'r') as dtb:
                inp = Input(shape=(4, dtb['qua_0_t'].shape[1]))
            self.build_CONV2D(inp, self.drp, arg)

        if self.cls['with_fea']:
            with h5py.File(self.inp, 'r') as dtb:
                inp = Input(shape=(dtb['fea_t'].shape[1], ))
            self.build_NDENSE(inp, self.drp, arg)

        if self.cls['with_fft']:
            with h5py.File(self.inp, 'r') as dtb:
                inp = Input(shape=(dtb['fft_a_t'].shape[1], ))
                self.build_NDENSE(inp, self.drp, arg)
                inp = Input(shape=(dtb['fft_g_t'].shape[1], ))
                self.build_NDENSE(inp, self.drp, arg)

        # Merge all the feature maps
        model = concatenate(self.models)
        sizes = np.linspace(2*len(self.classes), model._keras_shape[1], num=n_tail).astype('int')
        for ele in sizes[::-1]:
            model = Dense(ele, **arg)(model)
            model = BatchNormalization()(model)
            model = PReLU()(model)
            model = AdaptiveDropout(self.drp.prb, self.drp)(model)
        # Fusion in last softmax layer
        model = Dense(len(self.classes), activation='softmax', **arg)(model)

        return model           

    # Lauch the fit
    # ini_dropout refers to the initial dropout rate
    # patience refers to the early stopping round
    # max_epochs refers to the maximum amount of epochs
    # batch refers to the batch_size
    # n_tail refers to the amount of layers to deal with concatenated feature map
    def learn(self, ini_dropout=0.5, patience=7, max_epochs=100, batch=32, n_tail=5) :

        # Build the corresponding model
        model = self.build(ini_dropout, n_tail)
        # Compile the model
        model = Model(inputs=self.inputs, outputs=model)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # Implements the early stopping    
        arg = {'monitor': 'val_acc', 'mode': 'max'}
        early = EarlyStopping(min_delta=1e-5, patience=patience, **arg)
        check = ModelCheckpoint(self.mod, period=1, save_best_only=True, save_weights_only=True, **arg)
        shuff = DataShuffler(self.inp)

        # Fit the model
        his = model.fit_generator(self.train_generator('t', batch=batch),
                    steps_per_epoch=len(self.l_t)//batch, verbose=1, 
                    epochs=max_epochs, callbacks=[self.drp, early, check, shuff],
                    shuffle=True, validation_steps=len(self.l_e)//batch,
                    validation_data=self.valid_generator('e', batch=batch), 
                    class_weight=class_weight(self.l_t))

        # Serialize the training history
        with open(self.his, 'wb') as raw: pickle.dump(his, raw)
        # Memory efficiency
        del model, arg, early, check, shuff, his