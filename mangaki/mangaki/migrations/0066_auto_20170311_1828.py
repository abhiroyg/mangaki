# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-03-11 18:28
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mangaki', '0065_auto_20170306_1225'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='artist',
            name='first_name',
        ),
        migrations.RemoveField(
            model_name='artist',
            name='last_name',
        ),
    ]
