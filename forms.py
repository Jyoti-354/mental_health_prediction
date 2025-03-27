from django import forms

class PredictionForm(forms.Form):
    text = forms.CharField(label="Enter your text", widget=forms.Textarea)
