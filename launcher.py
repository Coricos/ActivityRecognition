from models import *

if __name__ == '__main__' :

    # mod = Models('XGBoost').learn(n_iter=75, verbose=0)
    # mod.save_model()
    # print(mod.performance())
    # del mod

    # mod = Models('RandomForest').learn(n_iter=75, verbose=0)
    # mod.save_model()
    # print(mod.performance())
    # del mod

    # mod = Models('Conv1D').learn(max_epochs=200, verbose=1)
    # mod.save_model()
    # print(mod.performance())
    # del mod

    # mod = Models('Conv2D').learn(max_epochs=200, verbose=1)
    # mod.save_model()
    # print(mod.performance())
    # del mod

    # mod = Models('DeepConv1D').learn(max_epochs=200, verbose=1)
    # mod.save_model()
    # print(mod.performance())
    # del mod

    mod = Models('DeepConv2D').learn(max_epochs=200, verbose=1)
    mod.save_model()
    print(mod.performance())
    del mod
