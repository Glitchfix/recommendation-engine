# pythonspot.com
from lightfm.evaluation import precision_at_k
from lightfm import LightFM
from flask import Flask, render_template, flash, request, redirect, url_for, session, flash
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, SelectField
import numpy as np

from lightfm.datasets import fetch_movielens

from rcengine import detect

data = fetch_movielens(min_rating=5.0)

print(repr(data['train']))
print(repr(data['test']))


model = LightFM(loss='warp')


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/", methods=['GET', 'POST'])
def hello():
    # form = ReusableForm(request.form)
    session.clear()

    # print(form.errors)
    if request.method == 'POST':
        l=int(request.form["pickUser"])
        X,Y = detect(l)
        flash([l,X,Y])
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
