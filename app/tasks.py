import pdftotext
import re

import nltk
from celery import shared_task
from django.conf import settings
from nltk.corpus import stopwords
from spellchecker import SpellChecker

from .models import Dokumen, Data, Pengujian


@shared_task
def simulate_sleep(length=5):
    import time
    time.sleep(length)
    return 'Finishing simulate sleep in {} second[s]'.format(length)


def calculate_f2(text, spell):
    tokens = nltk.word_tokenize(text, preserve_line=True)
    misspelled = spell.unknown(tokens)
    jumlah = len(misspelled)
    f2 = jumlah
    # for word in misspelled:
    # print("misspelled : " + word)
    # Get the one `most likely` answer
    # print("most likely : " + spell.correction(word))
    # Get a list of `likely` options
    # print("likely options : ")
    # print(spell.candidates(word))
    return f2


def to_text(path):
    with open(path, "rb") as f:
        pdf = pdftotext.PDF(f)
        text = "".join(pdf)
        return text


def calculate_feature_34(doc_id):
    from dlnn.tests.stringmatching.TestStringMatching import calculate
    # get target document
    target = to_text(Dokumen.objects.filter(id=doc_id).first().filenya.path)
    sources = []
    for i in range(1, 6):  # Acuan dokumen id 1 - 5
        sources.append(to_text(Dokumen.objects.filter(id=i).first().filenya.path))

    a1 = 0
    a2 = 0
    for source in sources:
        sc1, _ = calculate(target, source, 3, 200, ct=5e-1)
        sc2, _ = calculate(target, source, 5, 200, ct=5e-1)
        a1 += sc1
        a2 += sc2
        # kalo mau model rasio bisa pakai yang dibawah ini
        # accumulation += int(round(sc * 1.0 / lc))
    return a1, a2


def calculate_feature_56(doc_id):
    from dlnn.tests.stringmatching.TestStringMatching import calculate
    from dlnn.tests.stringmatching.TestWordScrapping import get_text
    from dlnn.tests.stringmatching.TestWordScrapping import repos
    # get target document
    target = to_text(Dokumen.objects.filter(id=doc_id).first().filenya.path)
    sources = []
    for i in range(1, 6):  # Acuan dokumen id 1 - 5
        text = get_text(repos[i])
        if text is not None:
            sources.append(text)

    a1 = 0
    a2 = 0
    for source in sources:
        sc1, _ = calculate(target, source, 3, 200, ct=5e-1)
        sc2, _ = calculate(target, source, 5, 200, ct=5e-1)
        a1 += sc1
        a2 += sc2
        # kalo mau model rasio bisa pakai yang dibawah ini
        # accumulation += int(round(sc * 1.0 / lc))
    return a1, a2


@shared_task(ignore_result=True)
def proceed_document(dokumen_id):
    import numpy
    from dlnn.Dlnn import Dlnn
    from dlnn.Dlnn import DLNN_DEFAULT_CONFIG
    dlnn = Dlnn(**DLNN_DEFAULT_CONFIG)
    # Todo : Load Dokumen by id (doc_id) [Dokumen.objects.filter(id=doc_id).first()]
    dokumen = Dokumen.objects.filter(id=dokumen_id).first()
    dokumen.state = "Process"
    dokumen.save()
    # Todo : Load pdf
    spell = SpellChecker()
    with open(dokumen.filenya.path, "rb") as f:
        pdf = pdftotext.PDF(f)
        text = "".join(pdf)

    # pecah kalimat menjadi kata kata
    text = text.lower()  # Converting to lowercase
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', text)  # Removing HTML tags
    sentence = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)  # Removing Punctuations

    tokens = nltk.word_tokenize(sentence, preserve_line=True)

    # Fitur 1 - cek salah ketik Bahasa Indonesia
    salah_ketik_indo = 0
    salah_ketik_english = 0
    url_stopword = set(stopwords.words('indonesian'))
    url_katadasar = settings.STATIC_ROOT + '/admin/db_text/kata-dasar-all.txt'
    url_stopword_en = set(stopwords.words('english'))

    # db_stopword = open(url_stopword,"r")
    db_katadasar = open(url_katadasar, "r")
    # db_stopword_en = open(url_stopword_en,"r")

    # stopword = db_stopword.read().split('\n')
    katadasar = db_katadasar.read().split('\n')
    # stopword_en = db_stopword_en.read().split('\n')

    for token in tokens:
        salah = True
        if (token in url_stopword):
            salah = False
        else:
            if token in katadasar:
                salah = False
        if salah:
            print("kata \"" + token + "\" salah")
            salah_ketik_indo += 1
        else:
            print("kata \"" + token + "\" betul")

    f1 = salah_ketik_indo
    dokumen.fitur1 = f1
    dokumen.save()

    # Todo : f2 = cari fitur 2 [calculate_feature_2()]
    for token in tokens:
        salah = True
        if (token in url_stopword_en):
            salah = False
        if salah:
            print("kata \"" + token + "\" salah")
            salah_ketik_english += 1
        else:
            print("kata \"" + token + "\" betul")

    # misspelled = spell.unknown(tokens)
    # jumlah = len(misspelled)
    f2 = salah_ketik_english
    dokumen.fitur2 = f2
    # for word in misspelled:
    #     print("misspelled : " + word)
    #     # Get the one `most likely` answer
    #     print("most likely : " + spell.correction(word))
    #     # Get a list of `likely` options
    #     print("likely options : ")
    #     print(spell.candidates(word))

    dokumen.save()

    f3, f4 = calculate_feature_34(dokumen_id)
    dokumen.fitur3 = f3
    dokumen.fitur4 = f4
    dokumen.save()

    f5, f6 = calculate_feature_56(dokumen_id)
    dokumen.fitur5 = f5
    dokumen.fitur6 = f6
    dokumen.save()

    # Todo : masukkan fitur f[1..4] ke database
    network = dlnn.get_model()
    result = network.predict(numpy.array([[f1, f2, f3, f4, f5, f6]]), batch_size=1)
    class_data = result.argmax(axis=1)[0]
    # print("Class Data {}".format(class_data))
    # Todo : masukkan class_data sebagai hasil kelas data [mappingkan dengan kelas seharusnya] [zero based indexing]
    dokumen.kualitas = class_data
    dokumen.state = "Done"
    dokumen.save()


@shared_task(ignore_result=True)
def testing_apps(gap_data):
    f1 = [[]]

    cek = Pengujian.objects.all()
    for a in cek:
        a.delete()
    dataset = Data.objects.filter(is_dataset=True)
    x = 0
    for data in dataset:
        x += 1
        print("data ke" + str(x))
        # Todo : Load pdf
        with open(data.url_file.path, "rb") as f:
            pdf = pdftotext.PDF(f)
            text = "".join(pdf)

        # Todo : Normalisasi
        # pecah kalimat menjadi kata kata
        text = text.lower()  # Converting to lowercase
        cleanr = re.compile('<.*?>')
        sentence = re.sub(cleanr, ' ', text)  # Removing HTML tags
        sentence = re.sub(r'[?|!|\'|"|#]', r'', sentence)
        sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)  # Removing Punctuations

        data_pdf = "".join(sentence)
        token_data_pdf = nltk.word_tokenize(data_pdf, preserve_line=True)

        # Fitur 1 - cek salah ketik Bahasa Indonesia
        url_dic_indo = settings.STATIC_ROOT + '/admin/db_text/kamus_indonesia.txt'
        kamus_indonesia = open(url_dic_indo, "r")
        katadasar = kamus_indonesia.read().split('\n')
        for i in range(len(katadasar)):
            katadasar[i] = katadasar[i].split("/")[0]

        salah_ketik_indo = 0
        for token in token_data_pdf:
            if token not in katadasar:
                salah_ketik_indo += 1

        # Fitur 2 - cek salah ketik Bahasa Inggris
        url_dic_en = settings.STATIC_ROOT + '/admin/db_text/kamus_english.txt'
        kamus_inggris = open(url_dic_en, "r")
        katadasar_en = kamus_inggris.read().split('\n')
        for i in range(len(katadasar_en)):
            katadasar_en[i] = katadasar_en[i].split("/")[0]

        salah_ketik_english = 0
        for token in token_data_pdf:
            if token not in katadasar_en:
                salah_ketik_english += 1

        akurasi_indo = int((len(token_data_pdf) - salah_ketik_indo) / len(token_data_pdf) * 100)
        akurasi_en = int((len(token_data_pdf) - salah_ketik_english) / len(token_data_pdf) * 100)

        new_hasil = Pengujian(perbandingan=str(x), fitur1=akurasi_indo, fitur2=akurasi_en)
        new_hasil.save()
        # if(cek == 0):
        #     new_hasil1 = Pengujian(perbandingan = "90:10", fitur1 = salah_ketik_indo, fitur2 = salah_ketik_english)
        #     new_hasil1.save()
        #     new_hasil2 = Pengujian(perbandingan = "80:20", fitur1 = salah_ketik_indo, fitur2 = salah_ketik_english)
        #     new_hasil2.save()
        #     new_hasil3 = Pengujian(perbandingan = "70:30", fitur1 = salah_ketik_indo, fitur2 = salah_ketik_english)
        #     new_hasil3.save()
        #     new_hasil4 = Pengujian(perbandingan = "60:40", fitur1 = salah_ketik_indo, fitur2 = salah_ketik_english)
        #     new_hasil4.save()
        #     new_hasil5 = Pengujian(perbandingan = "50:50", fitur1 = salah_ketik_indo, fitur2 = salah_ketik_english)
        #     new_hasil5.save()
        # else:
        #     hasil = Pengujian.objects.all()
        #     for data in hasil:
        #         data.fitur1 = salah_ketik_indo
        #         data.fitur2 = salah_ketik_english
        #         data.save()

    # # for x in range(jml_pengujian):
    # jml_uji = (100 - 50) / gap_data
    # dlatih = 100
    # duji = 0
    #
    # for y in range(int(jml_uji)):
    #     print("tes ke"+str(y))
    #     dlatih -= gap_data
    #     duji += gap_data
    #
    #     dataset = Data.objects.filter(is_dataset=True)
    #     jml_dataset = Data.objects.filter(is_dataset=True).count()
    #     jml_data_latih = int(jml_dataset * dlatih / 100)
    #     print("data latih"+str(jml_data_latih))
    #     i = 0
    #     for data in dataset:
    #         i += 1
    #         # Todo : Load pdf
    #         with open(data.url_file.path, "rb") as f:
    #             pdf = pdftotext.PDF(f)
    #             text = "".join(pdf)
    #
    #         # Todo : Normalisasi
    #         # pecah kalimat menjadi kata kata
    #         text = text.lower() # Converting to lowercase
    #         cleanr = re.compile('<.*?>')
    #         sentence = re.sub(cleanr, ' ', text)        #Removing HTML tags
    #         sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    #         sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence) #Removing Punctuations
    #
    #         if i <= jml_data_latih:
    #             datalatih = "".join(sentence)
    #         else:
    #             datauji = "".join(sentence)
    #
    #     token_datalatih = nltk.word_tokenize(datalatih, preserve_line=True)
    #     token_datauji = nltk.word_tokenize(datauji, preserve_line=True)
    #
    #     # Fitur 1 - cek salah ketik Bahasa Indonesia
    #     url_dic_indo = settings.STATIC_ROOT+'/admin/db_text/kamus_indonesia.txt'
    #     kamus_indonesia = open(url_dic_indo,"r")
    #     katadasar = kamus_indonesia.read().split('\n')
    #     for i in len(katadasar):
    #         katadasar[i] = katadasar[i].split("/")[0]
    #
    #     salah_ketik_indo = 0
    #     for token in token_datalatih:
    #         if token not in katadasar:
    #             salah_ketik_indo+=1
    #
    #     for token in token_datauji:
    #         if token not in katadasar:
    #             salah_ketik_indo+=1
    #
    #     # Fitur 2 - cek salah ketik Bahasa Inggris
    #     salah_ketik_english = 0
    #     url_dic_en = settings.STATIC_ROOT+'/admin/db_text/kamus_indonesia.txt'
    #     kamus_inggris = open(url_dic_en,"r")
    #     katadasar_en = kamus_inggris.read().split('\n')
    #     for i in len(katadasar_en):
    #         katadasar_en[i] = katadasar_en[i].split("/")[0]
    #
    #     salah_ketik_indo = 0
    #     for token in token_datalatih:
    #         if token not in katadasar_en:
    #             salah_ketik_english+=1
    #
    #     for token in token_datauji:
    #         if token not in katadasar_en:
    #             salah_ketik_english+=1
    #
    #     # for token in token_datauji:
    #     #     salah = True
    #     #     if(token in token_datalatih):
    #     #         salah = False
    #     #     if salah:
    #     #         print("kata \""+token+"\" salah")
    #     #         salah_ketik_indo += 1
    #     #     else:
    #     #         print("kata \""+token+"\" betul")
    #
    #     f1[y][0] = salah_ketik_indo
    #     f1[y][1] = salah_ketik_english
    #
    #
    # for x in f1:
    #     for y in x:
    #         print(str(y))
    #     print("\n")
