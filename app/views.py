import codecs
import logging
import pdftotext
import pickle
import datetime

import nltk
from django.contrib import messages
from django.contrib.auth import authenticate, login as do_login, logout
# from filetransfers.api import serve_file
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User, Group
from django.core.mail import send_mail
from django.db import IntegrityError
from django.db.models import Q
from django.http import BadHeaderError, Http404
from django.shortcuts import get_object_or_404, render, redirect
from django.utils.crypto import get_random_string
from django.utils.timezone import now
# from filetransfers.api import serve_file
# fitur
# from spellchecker import SpellChecker

from app.app.forms import auth, formKelas, formUploadDokumen, formUploadData
from app.app.utils.arrayutil import array_except, array_merge
from app.app.utils.commonutil import fetch_message, initialize_form_context
from app.app.utils.custom.decorators import login_required, auth_unneeded
from app.tasks import proceed_document, testing_apps
from .models import Dokumen, ResetPassword, Kelas, Data, Pengujian

logger = logging.getLogger('debug')


# == Landing Page ===============================================================================
@login_required(login_url='/login')
def index(request):
    date_now = datetime.datetime.today()
    new_docs = Dokumen.objects.filter(pub_date__year=date_now.year, pub_date__month=date_now.month,
                                      pub_date__day=date_now.day).count()
    all_docs = Dokumen.objects.all().count()
    users = User.objects.filter(is_staff=False, is_superuser=False).count()
    kelas = Kelas.objects.all().count()
    task = Dokumen.objects.filter(~Q(state="Done"))
    jumlah_task = Dokumen.objects.filter(~Q(state="Done")).count()
    context = array_merge(initialize_form_context(), fetch_message(request))
    context['menu'] = {
        'lv1': 'Dashboard',
    }
    context['data'] = {
        'new_docs': new_docs,
        'all_docs': all_docs,
        'users': users,
        'kelas': kelas,
        'task': task,
        'jumlah_task': jumlah_task,
    }
    return render(request, 'app/index.html', context)


# == Authentication ===============================================================================
@auth_unneeded()
def login(request):
    context = array_merge(initialize_form_context(), fetch_message(request))

    if request.method == 'POST':
        form = auth.Login(request.POST)
        context['form']['data'] = array_except(dict(form.data), ['csrfmiddlewaretoken'])
        if form.is_valid():
            user_data = authenticate(request,
                                     username=form.cleaned_data.get('email'),
                                     password=form.cleaned_data.get('password'))
            if user_data is not None:
                group = user_data.groups.first()
                if group is not None:
                    do_login(request, user_data)
                    from app.app.utils.sess_util import GROUP_ID
                    request.session[GROUP_ID] = group.id
                    callback = pickle.dumps(
                        {'message': {'notification': [{'msg': 'Login Success', 'level': 'success'}]}})
                    messages.add_message(request, messages.INFO, codecs.encode(callback, "base64").decode(), 'callback')
                    return redirect(
                        request.POST.get('next') if (
                                request.POST.get('next') and request.POST.get('next') != "") else '/user')
                else:
                    context['message']['notification'] = [{'msg': 'Your account is not activated yet', 'level': 'info'}]
                    return render(request, 'app/login.html', context)
            else:
                context['message']['notification'] = [{'msg': 'Account does not exists', 'level': 'info'}]
                return render(request, 'app/login.html', context)
        else:
            context['form']['errors'] = dict(form.errors)
            return render(request, 'app/login.html', context)
    else:
        if request.GET.get('next') and request.GET.get('next') != "":
            context['form']['data']['next'] = request.GET.get('next')
        return render(request, 'app/login.html', context)


@auth_unneeded()
def register(request):
    context = array_merge(initialize_form_context(), fetch_message(request))

    if request.method == 'POST':
        form = auth.Register(request.POST)
        context['form']['data'] = array_except(dict(form.data), ['csrfmiddlewaretoken'])
        if form.is_valid():
            if form.cleaned_data.get('password') != form.cleaned_data.get('password_conf'):
                context['form']['errors'] = {'password': 'Password Unequal', 'password_conf': 'Password Unequal'}
                return render(request, 'app/register.html', context)
            elif len(User.objects.filter(email=form.cleaned_data.get('email'))) > 0:
                context['message']['notification'] = [{'msg': 'Email exists', 'level': 'info'}]
                return render(request, 'app/register.html', context)
            else:
                group = Group.objects.filter(name=form.cleaned_data.get('role')).first()
                if group is not None:
                    account = User(username=form.cleaned_data.get('username'),
                                   email=form.cleaned_data.get('email'),
                                   password=make_password(form.cleaned_data.get('password')))
                    try:
                        account.save()
                        account.groups.add(group)
                    except IntegrityError:
                        context['message']['notification'] = [{'msg': 'Username is already taken', 'level': 'info'}]
                        return render(request, 'app/register.html', context)
                    callback = pickle.dumps({
                        'message': {'alert': [{'msg': 'Registration Success', 'level': 'success'}]},
                        'form': {'data': {'email': form.cleaned_data.get('email'), }}
                    })
                    messages.add_message(request, messages.INFO, codecs.encode(callback, "base64").decode(), 'callback')
                    return redirect('/login')
                else:
                    context['form']['errors'] = {'role': 'Role is invalid'}
                    return render(request, 'app/register.html', context)
        else:
            context['form']['errors'] = dict(form.errors)
            return render(request, 'app/register.html', context)
    else:
        return render(request, 'app/register.html', context)


@auth_unneeded()
def forgot(request):
    context = array_merge(initialize_form_context(), fetch_message(request))

    if request.method == 'POST':
        form = auth.Forgot(request.POST)
        context['form']['data'] = array_except(dict(form.data), ['csrfmiddlewaretoken'])
        context['form']['data']['forgot_concern'] = True
        if form.is_valid():
            account = User.objects.filter(email=form.cleaned_data.get('email'))
            if account.exists():
                account = account.first()
                token = get_random_string(length=80)
                if ResetPassword.objects.filter(user_id=account.id).exists():
                    ResetPassword.objects.filter(user_id=account.id).update(token=token)
                else:
                    ResetPassword.objects.create(token=token, created_at=now(), user_id=account.id)
                subject = 'Password Recover Request'
                message = """
                <doctype html>
                <html>
                    <head>
                    </head>
                    <body>
                        Here your password recover link address <a href="{base_url(request)}/recover?token={token}" target="_blank">Click</a>
                    </body>
                </html>
                """
                from_email = 'sume@noreply.com'
                if subject and message and from_email:
                    try:
                        send_mail(subject, message, from_email, [form.cleaned_data.get('email')], html_message=message)
                    except BadHeaderError:
                        context['message']['alert'] = [{'msg': 'Server Error, Try Again', 'level': 'danger'}]
                        return render(request, 'app/login.html', context)
                    context['message']['custom'] = {
                        'recover_success': 'Your recover form has been sent to your email account'}
                    context['form']['data']['email'] = ''
                    return render(request, 'app/login.html', context)
                else:
                    context['message']['alert'] = [{'msg': 'Server Error, Try Again', 'level': 'danger'}]
                    return render(request, 'app/login.html', context)
            else:
                context['message']['notification'] = [{'msg': 'Account does not exists', 'level': 'info'}]
                return render(request, 'app/login.html', context)
        else:
            context['form']['errors'] = dict(form.errors)
            return render(request, 'app/login.html', context)
    else:
        callback = pickle.dumps({
            'form': {
                'data': {
                    'forgot_concern': True,
                }
            }
        })
        messages.add_message(request, messages.INFO, codecs.encode(callback, "base64").decode(), 'callback')
        return redirect('/login')


@auth_unneeded()
def recover(request):
    context = array_merge(initialize_form_context(), fetch_message(request))
    token = request.GET.get('token') if request.GET.get('token') and request.GET.get('token') else request.POST.get(
        'token')
    if token and token != "":
        data = ResetPassword.objects.filter(token=token)
        if data.exists():
            recover_data = data.first()
        else:
            callback = pickle.dumps({'message': {'notification': [{'msg': 'Bad Token', 'level': 'warning'}]}})
            messages.add_message(request, messages.INFO, codecs.encode(callback, "base64").decode(), 'callback')
            return redirect('/login')
    else:
        callback = pickle.dumps({'message': {'notification': [{'msg': 'Token Not Provided', 'level': 'info'}]}})
        messages.add_message(request, messages.INFO, codecs.encode(callback, "base64").decode(), 'callback')
        return redirect('/login')

    if request.method == 'POST':
        form = auth.Recover(request.POST)
        context['form']['data'] = array_except(dict(form.data), ['csrfmiddlewaretoken'])
        if form.is_valid():
            if form.cleaned_data.get('password') != form.cleaned_data.get('password_conf'):
                context['form']['errors'] = {'password': 'Password Unequal', 'password_conf': 'Password Unequal'}
                return render(request, 'app/recover.html', context)
            else:
                account = User.objects.filter(id=recover_data.user_id).first()
                if account is not None:
                    account.password = make_password(form.cleaned_data.get('password'))
                    account.save()
                    callback = pickle.dumps({
                        'message': {'alert': [{'msg': 'Registration Success', 'level': 'success'}]},
                        'form': {'data': {'email': account.email}}
                    })
                    recover_data.delete()
                    messages.add_message(request, messages.INFO, codecs.encode(callback, "base64").decode(), 'callback')
                    return redirect('/login')
                else:
                    context['message']['alert'] = [{'msg': 'Account not found', 'level': 'danger'}]
                    return render(request, 'app/login.html', context)
        else:
            context['form']['errors'] = dict(form.errors)
            return render(request, 'app/recover.html', context)
    else:
        context['form']['data']['token'] = token
        return render(request, 'app/recover.html', context)


@login_required(login_url='/login')
def logout_view(request):
    callback = pickle.dumps({'message': {'notification': [{'msg': 'Logout Success', 'level': 'success'}]}})
    messages.add_message(request, messages.INFO, codecs.encode(callback, "base64").decode(), 'callback')
    logout(request)
    return redirect('/login')


# == User Management ===============================================================================
@login_required(login_url='/login')
def user(request, group_id=-1):
    context = array_merge(initialize_form_context(), fetch_message(request))

    groups = Group.objects.all()
    i = 0
    for group in groups:
        groups[i].count = group.user_set.filter(Q(is_staff=False) | Q(is_superuser=False)).count()
        i += 1

    if group_id == -1:
        users = User.objects.filter(Q(is_staff=False) | Q(is_superuser=False))
        i = 0
        for user in users:
            users[i].group = user.groups.all()
            i += 1
    else:
        group = Group.objects.get(pk=group_id)
        users = group.user_set.filter(Q(is_staff=False) | Q(is_superuser=False))
        i = 0
        for user in users:
            users[i].group = user.groups.all()
            i += 1
    context['data']['master'] = {
        'users': users,
        'groups': groups
    }
    context['menu'] = {
        'lv1': 'master',
        'lv2': 'master_user'
    }
    return render(request, 'app/user.html', context)


@login_required(login_url='/login')
def admin(request, mode_admin=-1):
    context = array_merge(initialize_form_context(), fetch_message(request))

    all = User.objects.filter(is_staff=True).count()
    staff = User.objects.filter(is_staff=True, is_superuser=False).count()
    superuser = User.objects.filter(is_superuser=True).count()

    if mode_admin == -1:
        users = User.objects.filter(Q(is_staff=True) | Q(is_superuser=True))
    elif mode_admin == 1:
        users = User.objects.filter(is_staff=True, is_superuser=False)
    elif mode_admin == 2:
        users = User.objects.filter(is_superuser=True)
    else:
        users = {}

    context['data']['master'] = {
        'users': users,
        'jumlah_staff': staff,
        'jumlah_superuser': superuser,
        'jumlah_semua': all
    }
    context['menu'] = {
        'lv1': 'master',
        'lv2': 'master_admin'
    }
    return render(request, 'app/admin.html', context)


# == Class Management ===============================================================================
@login_required(login_url='/login')
def kelas(request):
    context = array_merge(initialize_form_context(), fetch_message(request))

    latest_kelas_list = Kelas.objects.order_by('-end')[:5]
    i = 0
    for kelas in latest_kelas_list:
        latest_kelas_list[i].jumlahmember = kelas.members.count()
        latest_kelas_list[i].jumlahdokumen = kelas.dokumen.count()
        i += 1

    context['data']['kelas'] = {
        'list': latest_kelas_list,
    }
    context['menu'] = {
        'lv1': 'kelas',
        'lv2': 'kelas_list'
    }
    return render(request, 'app/kelas.html', context)


@login_required(login_url='/login')
def kelasbaru(request):
    context = array_merge(initialize_form_context(), fetch_message(request))
    context['menu'] = {
        'lv1': 'kelas',
        'lv2': 'kelas_newclass'
    }
    if request.method == 'POST':
        form = formKelas.BuatKelas(request.POST)
        context['form']['data'] = array_except(dict(form.data), ['csrfmiddlewaretoken'])
        if form.is_valid():
            name = form.cleaned_data.get('name')
            deskripsi = form.cleaned_data.get('deskripsi')
            members = form.cleaned_data.get('members')
            staffs = form.cleaned_data.get('staffs')
            startdate = form.cleaned_data.get('startdate')
            enddate = form.cleaned_data.get('enddate')

            new_kelas = Kelas(namakelas=name, keterangan=deskripsi, start=startdate, end=enddate)
            new_kelas.save()
            for member in members:
                anggota1 = User.objects.get(pk=member)
                new_kelas.members.add(anggota1)
            for staff in staffs:
                anggota2 = User.objects.get(pk=staff)
                new_kelas.members.add(anggota2)
            context['message']['notification'] = [{'msg': 'Pembuatan kelas berhasil', 'level': 'info'}]
            return render(request, 'app/kelasbaru.html', context)
        else:
            context['form']['errors'] = dict(form.errors)
            return render(request, 'app/kelasbaru.html', context)
    else:
        form = formKelas.BuatKelas()
        users = User.objects.filter(is_staff=False, is_superuser=False)
        staff = User.objects.filter(is_staff=True, is_superuser=False)
        context['data']['kelas'] = {
            'users': users,
            'staffs': staff,
        }
        context['form']['data'] = form

        return render(request, 'app/kelasbaru.html', context)


@login_required(login_url='/login')
def detailkelas(request, kelas_id):
    context = array_merge(initialize_form_context(), fetch_message(request))

    # dokumen = get_object_or_404(Dokumen, pk=question_id)
    # return serve_file(request, dokumen.filenya)
    kelas = Kelas.objects.get(pk=kelas_id)
    kelas.jumlahmember = kelas.members.count()
    kelas.jumlahdokumen = kelas.dokumen.count()
    documents = kelas.dokumen.all()
    context['data']['kelas'] = {
        'detail': kelas,
        'documents': documents
    }
    context['menu'] = {
        'lv1': 'kelas',
        'lv2': 'kelas_list'
    }
    return render(request, 'app/detail.html', context)


@login_required(login_url='/login')
def process_document(request, kelas_id):
    ids = Dokumen.objects.filter(kelas__id=kelas_id).values_list('id', flat=True)
    for idk in ids:
        proceed_document.delay(idk)
    return redirect('/kelas/{}/detail'.format(kelas_id))


@login_required(login_url='/login')
def editkelas(request, kelas_id):
    context = array_merge(initialize_form_context(), fetch_message(request))
    context['menu'] = {
        'lv1': 'kelas',
        'lv2': 'kelas_list'
    }
    if request.method == 'POST':
        form = formKelas.BuatKelas(request.POST)
        context['form']['data'] = array_except(dict(form.data), ['csrfmiddlewaretoken'])
        if form.is_valid():
            name = form.cleaned_data.get('name')
            deskripsi = form.cleaned_data.get('deskripsi')
            members = form.cleaned_data.get('members')
            staffs = form.cleaned_data.get('staffs')
            startdate = form.cleaned_data.get('startdate')
            enddate = form.cleaned_data.get('enddate')

            kelas = Kelas.objects.get(pk=kelas_id)
            kelas.namakelas = name
            kelas.keterangan = deskripsi
            kelas.start = startdate
            kelas.end = enddate
            # new_kelas = Kelas(namakelas=name, keterangan=deskripsi, start=startdate, end=enddate)
            kelas.save()
            for member in members:
                anggota1 = User.objects.get(pk=member)
                kelas.members.add(anggota1)
            for staff in staffs:
                anggota2 = User.objects.get(pk=staff)
                kelas.members.add(anggota2)
            context['form']['data'] = form
            context['message']['notification'] = [{'msg': 'Edit kelas berhasil', 'level': 'info'}]
            return render(request, 'app/edit_kelas.html', context)
        else:
            context['form']['errors'] = dict(form.errors)
            context['message']['notification'] = [{'msg': 'Edit kelas gagal', 'level': 'danger'}]
            return render(request, 'app/edit_kelas.html', context)
    else:
        form = formKelas.BuatKelas()
        users = User.objects.filter(is_staff=False, is_superuser=False)
        staff = User.objects.filter(is_staff=True, is_superuser=False)
        kls = Kelas.objects.get(pk=kelas_id)
        # members = User.objects.filter()
        # temp = []
        # for staf in staff:
        #     print(kls.namakelas)
        #     for mem in kls.members:
        #         print("tes")
        #         if staf.id in mem.id:
        #             print("ok")
        #             temp[0].status = True
        #             temp[0].staf = staf
        #             print("wah")
        #         else:
        #             print("tes")
        #             temp[1].status = False
        #             temp[1].staf = staf
        #             print("loh")
        #
        # print(temp)

        context['data']['kelas'] = {
            'kelas_id': kelas_id,
            'kls': kls,
            'users': users,
            'staffs': staff,
        }
        context['form']['data'] = form
    return render(request, 'app/edit_kelas.html', context)


@login_required(login_url='/login')
def upload_dokumen(request, kelas_id):
    context = array_merge(initialize_form_context(), fetch_message(request))
    context['menu'] = {
        'lv1': 'kelas',
        'lv2': 'kelas_list'
    }
    if request.method == 'POST':
        form = formUploadDokumen.UploadDokumen(request.POST, request.FILES)

        if form.is_valid():
            name = form.cleaned_data.get('name')
            filenya = form.cleaned_data.get('file_upload')
            current_user = request.user
            user = User.objects.get(pk=current_user.id)

            new_dokumen = Dokumen(user=user, nama_file=name, filenya=filenya)
            new_dokumen.save()

            kelas = Kelas.objects.get(pk=kelas_id)
            kelas.dokumen.add(new_dokumen)
            proceed_document.delay(new_dokumen.id)
            callback = pickle.dumps({
                'message': {'alert': [{'msg': 'Upload Success', 'level': 'success'}]},
            })
            messages.add_message(request, messages.INFO, codecs.encode(callback, "base64").decode(), 'callback')
            return redirect('/kelas/{}/detail'.format(kelas_id))
        else:
            context['form']['errors'] = dict(form.errors)
            return render(request, 'app/upload_dokumen.html', context)
    else:
        context['form']['data'] = formUploadDokumen.UploadDokumen()
        return render(request, 'app/upload_dokumen.html', context)


@login_required(login_url='/login')
def view_dokumen(request, dokumen_id, kelas_id=-1):
    if (kelas_id == -1):
        dokumen = get_object_or_404(Dokumen, pk=dokumen_id)
    else:
        kelas = get_object_or_404(Kelas, pk=kelas_id)
        dokumen = get_object_or_404(kelas.dokumen, pk=dokumen_id)

    return serve_file(request, dokumen.filenya)


@login_required(login_url='/login')
def hapus_kelas(request, kelas_id):
    context = array_merge(initialize_form_context(), fetch_message(request))
    context['menu'] = {
        'lv1': 'kelas',
        'lv2': 'kelas_list'
    }
    # hapus = Kelas.objects.delete(pk=kelas_id)
    hapus = False;
    if hapus:
        context['data']['notifikasi'] = {
            'noted': "Kelas berhasil dihapus",
            'type': 'success',
        }
    else:
        context['data']['notifikasi'] = {
            'noted': "Kelas gagal dihapus",
            'type': 'error',
        }
    latest_kelas_list = Kelas.objects.order_by('-end')[:5]
    i = 0
    for kelas in latest_kelas_list:
        latest_kelas_list[i].jumlahmember = kelas.members.count()
        latest_kelas_list[i].jumlahdokumen = kelas.dokumen.count()
        i += 1

    context['data']['kelas'] = {
        'list': latest_kelas_list,
    }
    return render(request, 'app/kelas.html', context)


@login_required(login_url='/login')
def statistik(request):
    context = array_merge(initialize_form_context(), fetch_message(request))
    context['menu'] = {
        'lv1': 'server',
        'lv2': 'server_statistik'
    }
    return render(request, 'app/statistik.html', context)


@login_required(login_url='/login')
def data_latih(request):
    context = array_merge(initialize_form_context(), fetch_message(request))
    datalatih = Data.objects.filter(datalatih=True)
    jdatalatih = Data.objects.filter(datalatih=True).count()
    context['menu'] = {
        'lv1': 'data',
        'lv2': 'data_latih'
    }
    context['dataset'] = {
        'data': datalatih,
        'jumlah': jdatalatih
    }
    return render(request, 'app/data.html', context)


@login_required(login_url='/login')
def data_uji(request):
    context = array_merge(initialize_form_context(), fetch_message(request))
    datauji = Data.objects.filter(datauji=True)
    jdatauji = Data.objects.filter(datauji=True).count()
    context['menu'] = {
        'lv1': 'data',
        'lv2': 'data_uji'
    }
    context['dataset'] = {
        'data': datauji,
        'jumlah': jdatauji
    }
    return render(request, 'app/data.html', context)


@login_required(login_url='/login')
def view_data(request, data_id):
    data = get_object_or_404(Data, pk=data_id)
    return serve_file(request, data.url_file)


@login_required(login_url='/login')
def data_upload(request):
    context = array_merge(initialize_form_context(), fetch_message(request))
    context['menu'] = {
        'lv1': 'data',
        'lv2': 'data_upload'
    }
    if request.method == 'POST':
        form = formUploadData.UploadData(request.POST, request.FILES)

        if form.is_valid():
            namafile = form.cleaned_data.get('namafile')
            datalatih = form.cleaned_data.get('datalatih')
            datauji = form.cleaned_data.get('datauji')
            is_dataset = form.cleaned_data.get('is_dataset')
            url_file = form.cleaned_data.get('url_file')

            new_data = Data(namafile=namafile, datalatih=datalatih, datauji=datauji, is_dataset=is_dataset,
                            url_file=url_file)
            new_data.save()

            callback = pickle.dumps({
                'message': {'alert': [{'msg': 'Upload Success', 'level': 'success'}]},
            })
            messages.add_message(request, messages.INFO, codecs.encode(callback, "base64").decode(), 'callback')
            return redirect('/data/upload')
        else:
            context['form']['errors'] = dict(form.errors)
            return render(request, 'app/upload_dataset.html', context)
    else:
        context['form']['data'] = formUploadData.UploadData()
        return render(request, 'app/upload_dataset.html', context)


@login_required(login_url='/login')
def edit_data(request, data_id):
    context = array_merge(initialize_form_context(), fetch_message(request))
    context['menu'] = {
        'lv1': 'data',
        'lv2': 'edit_data'
    }
    if request.method == 'POST':
        form = formUploadData.EditData(request.POST, request.FILES)

        if form.is_valid():
            namafile = form.cleaned_data.get('namafile')
            datalatih = form.cleaned_data.get('datalatih')
            datauji = form.cleaned_data.get('datauji')
            is_dataset = form.cleaned_data.get('is_dataset')
            url_file = form.cleaned_data.get('url_file')

            data = get_object_or_404(Data, pk=data_id)
            data.namafile = namafile
            data.datalatih = datalatih
            data.datauji = datauji
            data.is_dataset = is_dataset
            if url_file:
                data.url_file = url_file

            data.save()

            callback = pickle.dumps({
                'message': {'alert': [{'msg': 'Edit Success', 'level': 'success'}]},
            })
            messages.add_message(request, messages.INFO, codecs.encode(callback, "base64").decode(), 'callback')
            return redirect('/data/edit_data/{}'.format(data_id))
        else:
            context['form']['errors'] = dict(form.errors)
            return render(request, 'app/edit_dataset.html', context)
    else:
        data = get_object_or_404(Data, pk=data_id)
        context['dataset'] = {
            'data': data,
            'data_id': data_id
        }
        context['form']['data'] = formUploadData.EditData()
        return render(request, 'app/edit_dataset.html', context)

def pengujian(request):
    context = array_merge(initialize_form_context(), fetch_message(request))
    context['menu'] = {
        'lv1': 'data',
        'lv2': 'edit_data'
    }
    cek = Pengujian.objects.all().count()
    if(cek > 0):
        hasil = Pengujian.objects.all()
        context['hasil'] = {
            'pengujian': hasil,
        }
    return render(request, 'app/pengujian.html', context)

def proses_pengujian(request):
    context = array_merge(initialize_form_context(), fetch_message(request))
    context['menu'] = {
        'lv1': 'data',
        'lv2': 'edit_data'
    }
    jml_pengujian = 10
    gap_data = 10
    testing_apps.delay(gap_data)

    cek = Pengujian.objects.all().count()
    if(cek > 0):
        hasil = Pengujian.objects.all()
        context['hasil'] = {
            'pengujian': hasil,
        }
    return redirect('/pengujian')

# def proses_perhitungan_fitur2(dokumen_id):
#     dokumen = get_object_or_404(Dokumen, pk=dokumen_id)
#     spell = SpellChecker()
#     with open(dokumen.filenya.path, "rb") as f:
#         pdf = pdftotext.PDF(f)
#         text = "".join(pdf)
#         tokens = nltk.word_tokenize(text, preserve_line=True)
#         misspelled = spell.unknown(tokens)
#         jumlah = len(misspelled)
#         dokumen.fitur2 = jumlah
#         dokumen.save()
#         for word in misspelled:
#             print("misspelled : " + word)
#             # Get the one `most likely` answer
#             print("most likely : " + spell.correction(word))
#             # Get a list of `likely` options
#             print("likely options : ")
#             print(spell.candidates(word))
