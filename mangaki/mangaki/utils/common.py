from django.conf import settings
from mangaki.utils.chrono import Chrono
from sklearn.metrics import mean_squared_error

import pickle
import os.path


class RecommendationAlgorithm:
    def __init__(self):
        self.verbose = settings.RECO_ALGORITHMS_DEFAULT_VERBOSE
        self.chrono = Chrono(self.verbose)
        self.nb_users = None
        self.nb_works = None

    def get_backup_path(self, filename):
        if filename is None:
            filename = self.get_backup_filename()
        return os.path.join(settings.PICKLE_DIR, filename)

    def has_backup(self, filename=None):
        if filename is None:
            filename = self.get_backup_filename()
        return os.path.isfile(self.get_backup_path(filename))

    def save(self, filename):
        with open(self.get_backup_path(filename), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """
        This function raises FileNotFoundException if no backup exists.
        """
        with open(self.get_backup_path(filename), 'rb') as f:
            backup = pickle.load(f)
        return backup

    def set_parameters(self, nb_users, nb_works):
        self.nb_users = nb_users
        self.nb_works = nb_works

    def get_shortname(self):
        return 'algo'

    def get_backup_filename(self):
        return '%s.pickle' % self.get_shortname()

    def compute_rmse(self, y_pred, y_true):
        return mean_squared_error(y_true, y_pred) ** 0.5

    def all_errors(self, X_train, X_test, y_train, y_test):
        self.fit(X_train, y_train)
        y_pred_train = self.predict(X_train)
        print('Train minmax', min(y_pred_train), max(y_pred_train))
        print('Train error', self.compute_rmse(y_pred_train, y_train))
        y_pred_test = self.predict(X_test)
        print('Test minmax', min(y_pred_test), max(y_pred_test))
        print('Test error', self.compute_rmse(y_pred_test, y_test))

    def __str__(self):
        return '[%s]' % self.get_shortname().upper()
