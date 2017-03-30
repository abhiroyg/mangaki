# -*- coding: utf-8 -*-
# Generated by Django 1.10.6 on 2017-03-30 19:43
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Artist',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('first_name', models.CharField(blank=True, max_length=32, null=True)),
                ('last_name', models.CharField(max_length=32)),
            ],
        ),
        migrations.CreateModel(
            name='Announcement',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=128)),
                ('text', models.CharField(max_length=512)),
            ],
        ),
        migrations.CreateModel(
            name='Editor',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=33)),
            ],
        ),
        migrations.CreateModel(
            name='Studio',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=35)),
            ],
        ),
        migrations.CreateModel(
            name='Genre',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=17)),
            ],
        ),
        migrations.CreateModel(
            name='Page',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.SlugField()),
                ('markdown', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('is_shared', models.BooleanField(default=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('avatar_url', models.CharField(blank=True, default='', max_length=128, null=True)),
                ('mal_username', models.CharField(blank=True, default='', max_length=64, null=True)),
                ('nsfw_ok', models.BooleanField(default=False)),
                ('reco_willsee_ok', models.BooleanField(default=False)),
                ('score', models.IntegerField(default=0)),
                ('newsletter_ok', models.BooleanField(default=True)),
            ],
        ),
        migrations.CreateModel(
            name='Work',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=128)),
                ('source', models.CharField(blank=True, max_length=1044)),
                ('poster', models.CharField(max_length=128)),
                ('date', models.DateField(blank=True, null=True)),
                ('nsfw', models.BooleanField(default=False)),
                ('synopsis', models.TextField(blank=True, default='')),
            ],
        ),
        migrations.CreateModel(
            name='Anime',
            fields=[
                ('work_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='mangaki.Work')),
                ('composer', models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='composed', to='mangaki.Artist')),
                ('director', models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='directed', to='mangaki.Artist')),
                ('anime_type', models.TextField(default='', max_length=42)),
                ('author', models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='authored', to='mangaki.Artist')),
                ('editor', models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='mangaki.Editor')),
                ('genre', models.ManyToManyField(to='mangaki.Genre')),
                ('nb_episodes', models.TextField(default='Inconnu', max_length=16)),
                ('origin', models.CharField(choices=[('japon', 'Japon'), ('coree', 'Coree'), ('france', 'France'), ('chine', 'Chine'), ('usa', 'USA'), ('allemagne', 'Allemagne'), ('taiwan', 'Taiwan'), ('espagne', 'Espagne'), ('angleterre', 'Angleterre'), ('hong-kong', 'Hong Kong'), ('italie', 'Italie'), ('inconnue', 'Inconnue'), ('intl', 'International')], default='', max_length=10)),
                ('studio', models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='mangaki.Studio')),
                ('anidb_aid', models.IntegerField(default=0)),
            ],
            bases=('mangaki.work',),
        ),
        migrations.CreateModel(
            name='Rating',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('choice', models.CharField(choices=[('favorite', 'Mon favori !'), ('like', "J'aime"), ('dislike', "Je n'aime pas"), ('neutral', 'Neutre'), ('willsee', 'Je veux voir'), ('wontsee', 'Je ne veux pas voir')], max_length=8)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('work', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mangaki.Work')),
            ],
        ),
        migrations.CreateModel(
            name='Suggestion',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateTimeField(auto_now=True)),
                ('problem', models.CharField(choices=[('title', "Le titre n'est pas le bon"), ('poster', 'Le poster ne convient pas'), ('synopsis', 'Le synopsis comporte des erreurs'), ('author', "L'auteur n'est pas le bon"), ('composer', "Le compositeur n'est pas le bon"), ('double', 'Ceci est un doublon'), ('nsfw', "L'oeuvre est NSFW"), ('n_nsfw', "L'oeuvre n'est pas NSFW"), ('ref', 'Proposer une URL (myAnimeList, AniDB, Icotaku, VGMdb, etc.)')], default='ref', max_length=8, verbose_name='Partie concernée')),
                ('message', models.TextField(blank=True, verbose_name='Proposition')),
                ('is_checked', models.BooleanField(default=False)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('work', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mangaki.Work')),
            ],
        ),
        migrations.CreateModel(
            name='Neighborship',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('score', models.DecimalField(decimal_places=3, max_digits=8)),
                ('neighbor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='neighbor', to=settings.AUTH_USER_MODEL)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Manga',
            fields=[
                ('work_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='mangaki.Work')),
                ('vo_title', models.CharField(max_length=128)),
                ('editor', models.CharField(max_length=32)),
                ('origin', models.CharField(choices=[('japon', 'Japon'), ('coree', 'Coree'), ('france', 'France'), ('chine', 'Chine'), ('usa', 'USA'), ('allemagne', 'Allemagne'), ('taiwan', 'Taiwan'), ('espagne', 'Espagne'), ('angleterre', 'Angleterre'), ('hong-kong', 'Hong Kong'), ('italie', 'Italie'), ('inconnue', 'Inconnue'), ('intl', 'International')], max_length=10)),
                ('manga_type', models.TextField(blank=True, choices=[('seinen', 'Seinen'), ('shonen', 'Shonen'), ('shojo', 'Shojo'), ('yaoi', 'Yaoi'), ('sonyun-manhwa', 'Sonyun-Manhwa'), ('kodomo', 'Kodomo'), ('ecchi-hentai', 'Ecchi-Hentai'), ('global-manga', 'Global-Manga'), ('manhua', 'Manhua'), ('josei', 'Josei'), ('sunjung-sunjeong', 'Sunjung-Sunjeong'), ('chungnyun', 'Chungnyun'), ('yuri', 'Yuri'), ('dojinshi-parodie', 'Dojinshi-Parodie'), ('manhwa', 'Manhwa'), ('yonkoma', 'Yonkoma')], max_length=16)),
                ('genre', models.ManyToManyField(to='mangaki.Genre')),
                ('mangaka', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='drew', to='mangaki.Artist')),
                ('writer', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='wrote', to='mangaki.Artist')),
            ],
            bases=('mangaki.work',),
        ),
        migrations.CreateModel(
            name='SearchIssue',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateTimeField(auto_now=True)),
                ('title', models.CharField(max_length=128)),
                ('poster', models.CharField(blank=True, max_length=128, null=True)),
                ('mal_id', models.IntegerField(blank=True, null=True)),
                ('score', models.IntegerField(blank=True, null=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Recommendation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('target_user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='target_user', to=settings.AUTH_USER_MODEL)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('work', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mangaki.Work')),
            ],
        ),
        migrations.CreateModel(
            name='Pairing',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateTimeField(auto_now=True)),
                ('artist', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mangaki.Artist')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('work', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mangaki.Work')),
                ('is_checked', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Deck',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('category', models.CharField(max_length=32)),
                ('sort_mode', models.CharField(max_length=32)),
                ('content', models.CommaSeparatedIntegerField(max_length=42000)),
            ],
        ),
        migrations.CreateModel(
            name='Reference',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url', models.CharField(max_length=512)),
                ('work', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mangaki.Work')),
                ('suggestions', models.ManyToManyField(blank=True, to='mangaki.Suggestion')),
            ],
        ),
        migrations.CreateModel(
            name='Ranking',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('object_id', models.PositiveIntegerField()),
                ('score', models.FloatField()),
                ('nb_ratings', models.PositiveIntegerField()),
                ('nb_stars', models.PositiveIntegerField()),
                ('content_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='contenttypes.ContentType')),
            ],
        ),
        migrations.CreateModel(
            name='Top',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(auto_now_add=True)),
                ('category', models.CharField(choices=[('directors', 'Réalisateurs'), ('authors', 'Auteurs'), ('composers', 'Compositeurs')], max_length=10, unique_for_date='date')),
                ('contents', models.ManyToManyField(through='mangaki.Ranking', to='contenttypes.ContentType')),
            ],
        ),
        migrations.AddField(
            model_name='ranking',
            name='top',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mangaki.Top'),
        ),
        migrations.CreateModel(
            name='Album',
            fields=[
                ('work_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='mangaki.Work')),
                ('catalog_number', models.CharField(max_length=20)),
                ('vgmdb_aid', models.IntegerField(blank=True, null=True)),
                ('composer', models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='composer', to='mangaki.Artist')),
            ],
            bases=('mangaki.work',),
        ),
        migrations.CreateModel(
            name='Track',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=32)),
                ('album', models.ManyToManyField(to='mangaki.Album')),
            ],
        ),
    ]
