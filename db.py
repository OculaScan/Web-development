from flask_sqlalchemy import SQLAlchemy
import datetime as dt
from flask import Flask

app = Flask(__name__)
app.secret_key = "buat_secret_key_lebih_rumit"

# Konfigurasi SQLAlchemy
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "mysql://root:@localhost/flask_deteksipenyakitmata"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class UserModel(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(255))
    create_at = db.Column(
        db.String(50), default=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def __repr__(self):
        return f"<UserModel(name='{self.name}', email='{self.email}', password='{self.password}', create_at='{self.create_at}')>"


class HistoryModel(db.Model):
    __tablename__ = "history"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    file_name = db.Column(db.String(50))
    result = db.Column(db.String(50))
    score = db.Column(db.Float)
    create_at = db.Column(
        db.String(50), default=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def __repr__(self):
        return f"<HistoryModel(user_id='{self.user_id}', file_name='{self.file_name}', result='{self.result}', score='{self.score}', create_at='{self.create_at}')>"


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("Tables created successfully.")
