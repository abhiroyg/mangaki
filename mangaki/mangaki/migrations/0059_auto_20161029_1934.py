# -*- coding: utf-8 -*-
# Generated by Django 1.10.2 on 2016-10-29 19:34
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mangaki', '0058_merge'),
    ]

    operations = [
        migrations.RenameField(
            model_name='work',
            old_name='poster',
            new_name='ext_poster',
        ),
    ]
