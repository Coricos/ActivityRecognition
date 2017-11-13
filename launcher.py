from models import *

if __name__ == '__main__' :

    mod = Models('Conv1D', truncate=True).learn(max_epochs=200, verbose=1)
    mod.save_model()
    print(mod.performance())
    del mod

    mod = Models('Conv2D', truncate=True).learn(max_epochs=200, verbose=1)
    mod.save_model()
    print(mod.performance())
    del mod

    mod = Models('LSTM', truncate=True).learn(max_epochs=200, verbose=1)
    mod.save_model()
    print(mod.performance())
    del mod

    mod = Models('DeepConv1D', truncate=True).learn(max_epochs=200, verbose=1)
    mod.save_model()
    print(mod.performance())
    del mod

    mod = Models('DeepConv2D', truncate=True).learn(max_epochs=200, verbose=1)
    mod.save_model()
    print(mod.performance())
    del mod

    mod = Models('DeepLSTM', truncate=True).learn(max_epochs=200, verbose=1)
    mod.save_model()
    print(mod.performance())
    del mod

    mod = Models('XGBoost', truncate=True).learn(n_iter=75, verbose=0)
    mod.save_model()
    print(mod.performance())
    del mod

    mod = Models('RandomForest', truncate=True).learn(n_iter=75, verbose=0)
    mod.save_model()
    print(mod.performance())
    del mod
