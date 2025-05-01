import requests
import json
from fastapi import FastAPI, HTTPException, Depends, status, Security
import aiohttp
from fastapi.security import SecurityScopes, HTTPAuthorizationCredentials, HTTPBearer
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional
import jwt
from backend_secrets import *
from fastapi.security import HTTPBearer

# Импорт модулей
# from tests.schemas import *

# from schemas import *

# Настройка FastAPI
app = FastAPI(root_path="/api")

origins = [
    "https://хост_на_фронт_в_яндексе",
    "http://localhost:3000",
    "https://localhost:3000",
    "https://хост_на_домене"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # related to `origins`
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка базы данных
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Шифрование паролей
# хз используется ли
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# TODO: add more validation if needed


class QuestionResponse(BaseModel):
    question: str
    answer: str


class QuestionBatchResponse(BaseModel):
    batch: list


# Модель для запроса и ответа
class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    username: str
    password: str
    email: str











