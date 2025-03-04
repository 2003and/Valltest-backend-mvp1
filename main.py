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
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import jwt
import os
from datetime import datetime, timedelta
from app.backend_secrets import *
from fastapi.security import HTTPBearer
from jwt import PyJWKClient
import ssl

import requests
# import ast

# Настройка FastAPI
app = FastAPI(root_path="/api")

# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert.chain("/cert.crt", keyfile="/cert.key")

origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "https://185.32.84.201:3000",
    "https://localhost:3000",
    "https://10111897.xyz"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

# Настройка базы данных
DATABASE_URL = "sqlite:///./app/database_vault/test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Шифрование паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Таблицы
class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    # hashed_password = Column(String)
    email = Column(String, unique=True)
    # role = Column(String) #TODO: add roles and permissions functionality


class Subject(Base):
    __tablename__ = "subject"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)


class Theme(Base):
    __tablename__ = "theme"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    subject_id = Column(Integer, ForeignKey("subject.id", ondelete="CASCADE"))


class Problem(Base):
    __tablename__ = "problem"
    id = Column(Integer, primary_key=True, index=True)
    raw_data = Column(String) # , unique=True, index=True
    correct_answer = Column(Integer)
    theme_id = Column(Integer, ForeignKey("theme.id", ondelete="CASCADE"))


class Answer(Base):
    __tablename__ = "answer"
    id = Column(Integer, primary_key=True, index=True)
    problem_id = Column(Integer, ForeignKey("problem.id", ondelete="CASCADE"))
    answer_content = Column(String)


class UserAnswers(Base):
    __tablename__ = "user_answers"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"))
    problem_id = Column(Integer, ForeignKey("problem.id", ondelete="CASCADE"))
    answer_num = Column(Integer)

# Модель для запроса и ответа
class QuestionRequest(BaseModel):
    topic: str = "math"
    difficulty: str = "easy" # Уровень сложности: easy, medium, hard
    amount: int = 10

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

Base.metadata.create_all(bind=engine)

# JWT настройки
SECRET_KEY = jwt_secret  # Замените на свой секретный ключ
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Модель для запроса и ответа
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str

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

auth = VerifyToken()

# OAuth2
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
token_auth_scheme = HTTPBearer()

async def fetchCurrentUser(sub, token):
    async with aiohttp.ClientSession(headers={
        "Authorization": f'Bearer {token}'
    }) as session:
        async with session.get(f'https://{AUTH0_DOMAIN}/api/v2/users/{sub}') as response:
            # Read the response text
            return await response.json()


# # Функции работы с пользователями
# def get_user(db, username: str):
#     return db.query(User).filter(User.username == username).first()

# def create_user(db, user: UserCreate):
#     hashed_password = pwd_context.hash(user.password)
#     db_user = User(username=user.username, hashed_password=hashed_password, email=user.email)
#     db.add(db_user)
#     db.commit()
#     db.refresh(db_user)
#     return db_user

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(plain_password, hashed_password)

# def authenticate_user(db, username: str, password: str):
#     user = get_user(db, username)
#     if not user or not verify_password(password, user.hashed_password):
#         return None
#     return user

# def create_access_token(data: dict, expires_delta: timedelta = None):
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=15)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# Эндпоинт для логина
# @app.get("/login", response_model=Token)
# async def login(token: str = Depends(token_auth_scheme)):
#     db = SessionLocal()
#     user = authenticate_user(db, token.username, token.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
#     return {"access_token": access_token, "token_type": "bearer"}

# Эндпоинт для защищенных ресурсов
@app.get("/users/me")
async def read_users_me(auth_result = Security(auth.verify)):
    print('Вызван защищённый авторизацией эндпоинт')
    print(json.dumps(auth_result, indent=4))
    # credentials_exception = HTTPException(
    #     status_code=status.HTTP_401_UNAUTHORIZED,
    #     detail="Could not validate credentials",
    #     headers={"WWW-Authenticate": "Bearer"},
    # )
    # try:
    #     result = token.credentials
    #     # payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    #     # username: str = payload.get("sub")
    #     # if username is None:
    #     #     raise credentials_exception
    #     # token_data = TokenData(username=username)
    # except jwt.PyJWTError:
    #     raise credentials_exception
    # db = SessionLocal()
    # user = get_user(db, username=token_data.username)
    # if user is None:
    #     raise credentials_exception
    # return user
    return auth_result


@app.get("/private")
def private(token: str = Depends(token_auth_scheme)):
    """A valid access token is required to access this route"""

    # result = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    result = jwt.decode(token.credentials, options={"verify_signature": False})

    return result

@app.post("/register/")
#async def register_user(username: str, password: str, email: str):
async def register_user(username: str, email: str):
    """
    Регистрация нового пользователя.
    """
    db = SessionLocal()
    try:
        # Проверяем, существует ли пользователь
        if get_user(username, db):
            raise HTTPException(status_code=400, detail="Пользователь уже существует.")

        # Создаем нового пользователя
        #hashed_password = pwd_context.hash(password)
        # new_user = User(username=username, hashed_password=hashed_password, email=email)
        new_user = User(username=username, email=email, )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"username": new_user.username, "message": "Пользователь успешно зарегистрирован."}
    finally:
        db.close()


@app.get("/my_answers/")
def get_user_answers_count(token: str = Depends(token_auth_scheme)):
    db = SessionLocal()
    # Проверяем, зарегистрирован ли пользователь
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    token_data = TokenData(username=username)
    user = authenticate_user(db, token_data.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Получаем общее количество ответов пользователя
    total_answers = db.query(UserAnswers).filter(UserAnswers.user_id == user.id).count()

    # Получаем количество правильных ответов
    correct_answers = db.query(UserAnswers).join(Problem).filter(UserAnswers.user_id == user.id, UserAnswers.answer_num == Problem.correct_answer, UserAnswers.problem_id==Problem.id).count()

    return {
        "total_answers": total_answers,
        "correct_answers": correct_answers
    }


@app.post("/add_answer/", response_model=BaseResponse)
async def add_answer(answer: AnswerRequest, token: str = Depends(token_auth_scheme)):
    db = SessionLocal()
    try:
        #user = authenticate_user(db, credentials.username, credentials.password)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
        user = authenticate_user(db, token_data.username)
        if not user:
            raise HTTPException(status_code=401, detail="Неверные учетные данные.")
        new_answer = UserAnswers(user_id=user.id, problem_id=answer.problem_num, answer_num=answer.answer_num)
        db.add(new_answer)
        db.commit()
        db.refresh(new_answer)
        return {"response": "ok!"}
    finally:
        db.close()


# Маршрут для генерации математических вопросов
@app.post("/generate_question/")# , response_model=QuestionBatchResponse)
async def generate_question(request: QuestionRequest):
#async def generate_question(request: str):
    """
    Генерирует математический вопрос и ответ с использованием OpenAI API.
    """
#    request = json.loads(request_raw)
    db = SessionLocal()
    # TODO: add latex parsing on front
    topics = {
        "integral": "Вопрос про интегралы, он должен быть либо про неопределённый интеграл, либо про определённый интеграл.",
        "matrix": "Вопрос про матрицы, даны квадратные матрицы A и B, необходимо выполнить арифметические операции A+B, A-B или A*B. Ответом должен быть результат этой арифметической операции", # "Вопрос про матрицы, там должны быть сложение матриц, умножение матриц, нахождение определителя матрицы и решение системы уравнений по правилу крамера",
        "pro": "Вопрос про производные, он должен быть про нахождение производной первого или второго порядка - 50\% шанс на кажудю", # derivatives
        "vector": "Вопрос про вектора, он должен быть про математическую операцию с векторами: либо сложение, либо вычитание, либо скалярное умножение, либо векторное умножение",
        "limit": "Вопрос про пределы, он должен быть про нахождение предела какой-либо функции",
    }
    try:
        # TODO: rewrite prompt to generate a JSON
        new_batch = []
        for i in range(request.amount):
        #TODO: add a dictionary/json with specific prompts for specific topics
        #This prompt will only be used if a specific prompt wasn't found in the aformentioned json
            temp = topics.get(request.topic.lower(), f"Тема вопроса - {request.topic}.")
            prompt = {
            "modelUri": "gpt://b1gefo1sef4nbt0kc8tb/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                # "maxTokens": "2000"
            },
            "messages": [
                {
                    "role": "user",
                    "text": f"Сгенерируй вопрос по математике {request.difficulty} сложности и скажи ответ. " +
                            temp +
                            "Напиши \"Ответ:\" перед ответом и не пиши \"Вопрос:\" перед вопросом." +
                            "Не объясняй как ты получил ответ, просто скажи его" +
                            "Используй случайные числа, не только 7 и 5, и не повторяйся" +
                            "Пиши \"$$\" в начале и в конце формулы, чтобы её можно было обработать парсером LaTeX"
                            # "Напиши формулы вопроса и ответа в LaTeX"

                    }
                ]
            }
            # prompt = (
            #     f"Generate a math question of {request.difficulty} difficulty and provide the solution. " +
            #     temp +
            #     "Write \"Answer:\" before the solution and don't write \"Question:\" before the question" +
            #     "Don't explain the answer, just tell me the result"
            #     "Use random numbers, not just 7 and 5, and don't repeat yourself"
            #     "Put \"$$\" in the beginning and end of the formula, so that it can be read by a LaTeX parser"
            # )
            print("Response generating....")
            # Запрос к YandexGPT
            url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Api-Key AQVN3Gy8RXHx2J4ocgbGY47qr9UMigMSI2nwDJDF"
            }
            response = requests.post(url, headers=headers, json=prompt)
            # response = client.chat.completions.create(
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": prompt,
            #         }
            #     ],
            #     model="gpt-4o-mini",
            #    max_tokens=150
            # )
            print(f"Answer {i} parsing....")
            # Извлечение ответа
            # content = response.choices[0].message.content.strip()
            content_raw = response.text
            content = json.loads(content_raw)["result"]["alternatives"][0]["message"]["text"]
            print(content)
            # content = ast.literal_eval(content_raw)

            # Разделение на вопрос и ответ
            if "Ответ:" in content:
                 question, answer = content.replace("Вопрос: ", "").split("Ответ:")
            else:
                 raise HTTPException(status_code=500, detail="Не удалось извлечь ответ.")
            new_batch.append({"question": question.strip(), "answer": answer.strip()})
            #return {"question": question.strip(), "answer": answer.strip()}
        print("returning")
        # return {"question": question.strip(), "answer": answer.strip()}

        return {"batch": new_batch}
        #print(json.loads(content.replace("\\",""))["result"]["alternatives"][0]["message"]["text"])
        # return {"batch": content}


        # raw_problem = {"question": "What is 2+3?", "answer": "5"}

        # # add record to "Problem" table
        # new_problem = Problem(theme_id=1, raw_data=raw_problem["question"], correct_answer=1)
        # db.add(new_problem)
        # db.commit()
        # db.refresh(new_problem)

        # # add record to "Answers" table
        # new_answer = Answer(problem_id=new_problem.id, answer_content=raw_problem["answer"])
        # db.add(new_answer)
        # db.commit()
        # db.refresh(new_answer)

        # ans = []
        # for i in range(request.amount):
        #     ans.append({"question": "What is 2+3?", "answer": "5"})
        # return {"batch": ans}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации вопроса: {e}")

# if __name__=="__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, ssl=ssl_context)
