from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
from django.db import models


# Create your models here.

class Dokumen(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    nama_file = models.CharField(max_length=200)
    pub_date = models.DateTimeField(verbose_name='date published')
    filenya = models.FileField(upload_to='dokumen/%Y/%m/%d/',
                               validators=[FileExtensionValidator(allowed_extensions=['pdf'])])

    def __str__(self):
        return self.nama_file


class ResetPassword(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=80)
    created_at = models.DateTimeField()

class Kelas(models.Model):
    namakelas = models.CharField(max_length=200)
    keterangan = models.TextField()
    members = models.ManyToManyField(User)
    start = models.DateTimeField()
    end = models.DateTimeField()
