from mangaki.utils.wals import MangakiWALS
from mangaki.utils.als import MangakiALS
from mangaki.utils.knn import MangakiKNN
from mangaki.utils.svd import MangakiSVD
from django.core.management.base import BaseCommand
from mangaki.utils.algo import fit_algo
from mangaki.models import Rating


ALGOS = {
    'knn': lambda: MangakiKNN(),
    'svd': lambda: MangakiSVD(20),
    'als': lambda: MangakiALS(20),
    'wals': lambda: MangakiWALS(20),
}


class Command(BaseCommand):
    args = ''
    help = 'Train a recommendation algorithm'

    def add_arguments(self, parser):
        parser.add_argument('algo_name', type=str)

    def handle(self, *args, **options):
        algo_name = options.get('algo_name')
        triplets = Rating.objects.values_list('user_id', 'work_id', 'choice')
        fit_algo(ALGOS[algo_name](), triplets)
        self.stdout.write(self.style.SUCCESS('Successfully fit %s' % algo_name))
