from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False, default=30)
    gender = db.Column(db.String(10), nullable=False, default='Other')
    phone_no = db.Column(db.String(15), unique=True, nullable=False)
    medical_history = db.Column(db.Text, nullable=True)

    # Emergency Contact
    emergency_contact_name = db.Column(db.String(100), nullable=False, default='Emergency Contact')
    emergency_contact_phone = db.Column(db.String(15), nullable=False, default='+1-555-0188')

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'full_name': self.name,
            'age': self.age,
            'gender': self.gender,
            'phone_no': self.phone_no,
            'medical_history': self.medical_history,
            'emergency_contact_name': self.emergency_contact_name,
            'emergency_contact_phone': self.emergency_contact_phone
        }

class SensorData(db.Model):
    __tablename__ = 'sensor_data'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    air_quality = db.Column(db.Integer, nullable=False)  # AQI
    pm25 = db.Column(db.Float, nullable=False)  # PM2.5
    so2_level = db.Column(db.Float, nullable=False)  # SO2 level
    no2_level = db.Column(db.Float, nullable=False)  # NO2 level
    co2_level = db.Column(db.Float, nullable=False)  # CO2 level
    humidity = db.Column(db.Float, nullable=False)  # Humidity
    temperature = db.Column(db.Float, nullable=False)  # Temperature

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            'air_quality': self.air_quality,
            'AQI': self.air_quality,
            'pm25': self.pm25,
            'PM2.5': self.pm25,
            'so2_level': self.so2_level,
            'no2_level': self.no2_level,
            'co2_level': self.co2_level,
            'humidity': self.humidity,
            'temperature': self.temperature
        }

class Alert(db.Model):
    __tablename__ = 'alert'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'message': self.message,
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp)
        }

class QuizResponse(db.Model):
    __tablename__ = 'quiz_response'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.String(255), nullable=False)  # Store full question text
    answer = db.Column(db.String(255), nullable=False)  # Store full selected answer

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'question': self.question,
            'answer': self.answer
        }
