from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import random
from head import db

class DataTraining(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    def __init__(self, **kwargs):
        super(DataTraining, self).__init__(**kwargs)
        for key, value in kwargs.items():
            if key != "id":
                setattr(self, key, value)

    def get_column_names(self):
        return [column.name for column in self.__table__.columns]

    def get_table_name(self):
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%Y%m%d")
        current_time = current_datetime.strftime("%H%M")
        random_digits = str(random.randint(10, 99))
        table_name = f"dt_v{current_date}{current_time}{random_digits}"
        return table_name

class ModelVersi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    versi_model = db.Column(db.String(255))
