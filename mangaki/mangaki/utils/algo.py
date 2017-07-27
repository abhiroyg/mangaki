from mangaki.utils.data import Dataset


def fit_algo(algo, triplets, titles=None, categories=None):
    dataset = Dataset()

    if titles is not None:
        dataset.titles = dict(titles)
    if categories is not None:
        dataset.categories = dict(categories)
    anonymized = dataset.make_anonymous_data(triplets)
    algo.set_parameters(anonymized.nb_users, anonymized.nb_works)
    algo.fit(anonymized.X, anonymized.y)
    if algo.get_shortname() in {'svd', 'als', 'knn'}:
        algo.save(algo.get_backup_filename())
        dataset.save('ratings-' + algo.get_backup_filename())
    return dataset, algo

def get_algo_backup(algo):
    algo.load(algo.get_backup_filename())
    return algo

def get_dataset_backup(algo):
    dataset = Dataset()
    dataset.load('ratings-' + algo.get_backup_filename())
    return dataset
