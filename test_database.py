# from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# DATABASE_URL = "sqlite:///./face_recognition.db"

# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# class FaceEmbedding(Base):
#     __tablename__ = "face_embeddings"
    
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, unique=True)
#     embedding = Column(LargeBinary)  # Storing as binary

# # Create Tables
# Base.metadata.create_all(bind=engine)


from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient
import bson

# SQLite Setup
DATABASE_URL = "sqlite:///./face_recognition.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# MongoDB Setup
MONGO_URI = "mongodb://localhost:27017/python"
client = MongoClient(MONGO_URI)
db = client["face_recognition"]
face_embeddings_collection = db["face_embeddings"]

# Migrate Data
for face in session.query(FaceEmbedding).all():
    face_embeddings_collection.insert_one({
        "user_id": face.user_id,
        "embedding": bson.Binary(face.embedding)
    })

print("Migration completed!")
