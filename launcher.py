from models import *

if __name__ == '__main__' :

    mod = Models('XGBoost').learn(n_iter=75, verbose=1)
    mod.save_model()
    del mod

    mod = Models('RandomForest').learn(n_iter=75, verbose=1)
    mod.save_model()
    del mod

