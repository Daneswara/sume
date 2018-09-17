import codecs
import logging
import pickle

from django.contrib import messages
from django.contrib.auth import authenticate, login as do_login, logout
# from filetransfers.api import serve_file
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User, Group
from django.db import IntegrityError
from django.http import HttpResponse, BadHeaderError, Http404
from django.shortcuts import get_object_or_404, render, redirect
from django.utils.crypto import get_random_string
from django.utils.timezone import now
from django.db.models import Q

from app.app.forms import auth
from app.app.utils.arrayutil import array_except, array_merge
from app.app.utils.commonutil import fetch_message, initialize_form_context, base_url
from app.app.utils.custom.decorators import login_required, auth_unneeded
from .forms import LoginForm
from .models import Dokumen, ResetPassword

logger = logging.getLogger('debug')


# from filetransfers.api import serve_file

@login_required(login_url='/login')
def index(request):
    context = {}
    # raise Http404("Poll does not exist")
    return render(request, 'app/index.html', context)

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
                                request.POST.get('next') and request.POST.get('next') != "") else ('/' + group.name))
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


def logout_view(request):
    logout(request)
    return render(request, 'app/login.html')

def dologin(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            user = authenticate(request, username=form.cleaned_data['username'], password=form.cleaned_data['password'])
            if user is not None:
                login(request, user)
                data = {
                    'username': user.username,
                }
                return redirect('/')
            else:
                return render(request, 'app/login.html', {'form': form})
                # Return an 'invalid login' error message.
                # Return an 'invalid login' error message.
    else:
        form = LoginForm()
    return render(request, 'app/login.html', {'form': form})


@login_required(login_url='/login')
def kelas(request):
    latest_dokumen_list = Dokumen.objects.order_by('-pub_date')[:5]
    context = {
        'latest_dokumen_list': latest_dokumen_list,
    }
    return render(request, 'app/kelas.html', context)


@login_required(login_url='/login')
def user(request, group_id=-1):
    groups = Group.objects.all()
    i=0
    for group in groups:
        groups[i].count = group.user_set.count()
        i+=1

    if group_id == -1:
        users = User.objects.all()
        i = 0
        for user in users:
            users[i].group = user.groups.all()
            i+=1
    else:
        group = Group.objects.get(pk = group_id)
        users = group.user_set.all()
        i = 0
        for user in users:
            users[i].group = user.groups.all()
            i+=1
    context = {
        'users': users,
        'groups': groups
    }
    return render(request, 'app/user.html', context)


@login_required(login_url='/login')
def admin(request, mode_admin = -1):

    staff = User.objects.filter(is_staff=True).count()
    superuser = User.objects.filter(is_superuser=True).count()

    if mode_admin == -1:
        users = User.objects.filter(Q(is_staff=True) | Q(is_superuser=True))
    elif mode_admin == 1:
        users = User.objects.filter(is_staff=True, is_superuser = False)
    elif mode_admin == 2:
        users = User.objects.filter(is_superuser=True)
    else:
        users = {}

    context = {
        'users': users,
        'jumlah_staff':staff,
        'jumlah_superuser':superuser
    }
    return render(request, 'app/admin.html', context)


@login_required(login_url='/login')
def detail(request, question_id):
    # dokumen = get_object_or_404(Dokumen, pk=question_id)
    # return serve_file(request, dokumen.filenya)
    latest_dokumen_list = Dokumen.objects.order_by('-pub_date')[:5]
    context = {
        'latest_dokumen_list': latest_dokumen_list,
    }
    return render(request, 'app/detail.html', context)



def results(request, question_id):
    response = "You're looking at the results of question %s."
    return HttpResponse(response % question_id)


def vote(request, question_id):
    return HttpResponse("You're voting on question %s." % question_id)


def openfile(request, question_id):
    filename = Dokumen.filenya.name.split('/')[-1]
    response = HttpResponse(object_name.file, content_type='text/plain')
    response['Content-Disposition'] = 'attachment; filename=%s' % filename

    return response

@login_required(login_url='/login')
def statistik(request):
    context = {
    }
    return render(request, 'app/statistik.html', context)
