from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
from django.db import models


# Create your models here.

class Dokumen(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    nama_file = models.CharField(max_length=200)
    pub_date = models.DateTimeField(auto_now_add=True)
    filenya = models.FileField(upload_to='dokumen/%Y/%m/%d/',
                               validators=[FileExtensionValidator(allowed_extensions=['pdf', 'doc', 'docx'])])
    plagiat = models.IntegerField(default=0)
    typo = models.IntegerField(default=0)
    fitur1 = models.IntegerField(default=0)
    fitur2 = models.IntegerField(default=0)
    fitur3 = models.IntegerField(default=0)
    fitur4 = models.IntegerField(default=0)
    fitur5 = models.IntegerField(default=0)
    fitur6 = models.IntegerField(default=0)
    kualitas = models.IntegerField(default=0)
    state = models.CharField(default="In Queue", max_length=100)

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
    dokumen = models.ManyToManyField(Dokumen)

class Data(models.Model):
    namafile = models.CharField(max_length=200)
    datalatih = models.BooleanField(default=True)
    datauji = models.BooleanField(default=False)
    is_dataset = models.BooleanField(default=True)
    url_file = models.FileField(upload_to='data/%Y/%m/%d/',
                               validators=[FileExtensionValidator(allowed_extensions=['pdf'])])

class Pengujian(models.Model):
    perbandingan = models.CharField(max_length=200, default="100%")
    fitur1 = models.IntegerField(default=0)
    fitur2 = models.IntegerField(default=0)
    fitur3 = models.IntegerField(default=0)
    fitur4 = models.IntegerField(default=0)
    fitur5 = models.IntegerField(default=0)
