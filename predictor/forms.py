from django import forms
from datetime import date

class ProductivityForm(forms.Form):
    date = forms.DateField(
        widget=forms.SelectDateWidget(years=range(2015, date.today().year + 1)),
        label='Date',
    )
    department = forms.ChoiceField(
        choices=[('sewing', 'Sewing'), ('finishing', 'Finishing')],
        label='Department',
    )
    team = forms.ChoiceField(
        choices=[(i, f'Team {i}') for i in range(1, 13)],
        label='Team',
    )
    
    targeted_productivity = forms.FloatField(min_value=0, max_value=1, label='Targeted Productivity',)
    std_minute_value = forms.FloatField(min_value=0, label='Standard Minute Value')
    work_in_progress = forms.IntegerField(min_value=0, label='Work in Progress')
    over_time = forms.IntegerField(min_value=0, label='Over Time')
    incentive = forms.FloatField(min_value=0, label='Incentive')
    idle_time = forms.FloatField(min_value=0, label='Idle Time')
    idle_men = forms.IntegerField(min_value=0, label='Idle Men')
    no_of_style_change = forms.IntegerField(min_value=0, label='Number of Style Changes')
    no_of_workers = forms.IntegerField(min_value=1, label='Number of Workers')


    def __init__(self, *args, **kwargs):
        super(ProductivityForm, self).__init__(*args, **kwargs)
        placeholders = {
            'targeted_productivity': 'e.g. 0.8',
            'std_minute_value': 'e.g. 26.16',
            'work_in_progress': 'e.g. 1108',
            'over_time': 'e.g. 7080',
            'incentive': 'e.g. 98',
            'idle_time': 'e.g. 0',
            'idle_men': 'e.g. 0',
            'no_of_style_change': 'e.g. 0',
            'no_of_workers': 'e.g. 59'
        }
        for field_name, placeholder in placeholders.items():
            self.fields[field_name].widget.attrs.update({'placeholder': placeholder})