from django import forms


class Login(forms.Form):
    email = forms.EmailField()
    password = forms.CharField()


class Register(forms.Form):
    username = forms.CharField()
    email = forms.EmailField()
    password = forms.CharField()
    password_conf = forms.CharField()
