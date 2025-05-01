# import requests
# import json
from fastapi import FastAPI, HTTPException, Depends, status, Security
# import aiohttp
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
# from tests import *
from src.tests.routes import test_router

# from schemas import *

# Настройка FastAPI
app = FastAPI(root_path="/")
app.include_router(test_router)#, prefix="")

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


class BaseResponse(BaseModel):
    response: str


class QuestionResponse(BaseModel):
    question: str
    answer: str


class AnswerRequest(BaseModel):
    problem_num: int
    answer_num: int
    correct_answer_num: int


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


# Создание таблиц
Base.metadata.create_all(bind=engine)


class UnauthorizedException(HTTPException):
    def __init__(self, detail: str, **kwargs):
        """Returns HTTP 403"""
        super().__init__(status.HTTP_403_FORBIDDEN, detail=detail)

class UnauthenticatedException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Requires authentication"
        )


class VerifyToken:
    """Does all the token verification using PyJWT"""

    def __init__(self):
        # This gets the JWKS from a given URL and does processing so you can
        # use any of the keys available
        self.jwks_client = jwt.PyJWKClient(JWKS_URL)

    async def verify(self,
                     security_scopes: SecurityScopes,
                     token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer())
                     ):
        if token is None:
            raise UnauthenticatedException

        # This gets the 'kid' from the passed token
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(
                token.credentials
            ).key
        except jwt.exceptions.PyJWKClientError as error:
            raise UnauthorizedException(str(error))
        except jwt.exceptions.DecodeError as error:
            raise UnauthorizedException(str(error))

        try:
            payload = jwt.decode(
                token.credentials,
                signing_key,
                audience=AUTH0_API_AUDIENCE,
                issuer=AUTH0_ISSUER,
                algorithms=AUTH0_ALGORITHMS,
            )
        except Exception as error:
            raise UnauthorizedException(str(error))

        user = await fetchCurrentUser(payload['sub'], token.credentials)

        return {
            "token_payload": payload,
            "user": user
        }





