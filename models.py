from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class DataTraining(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    versi = db.Column(db.String(100))
    response = db.Column(db.String(255))
    level = db.Column(db.String(255))

class ModelVersi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    versi_model = db.Column(db.String(255))
