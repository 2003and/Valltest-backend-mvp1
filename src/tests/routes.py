from fastapi import APIRouter, HTTPException, Depends, status, Security
from fastapi.security import SecurityScopes, HTTPAuthorizationCredentials, HTTPBearer
import duckdb
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Optional

import requests
# import aiohttp
import jwt
import json
import random

from backend_secrets import *
from src.tests.schemas import *


test_router = APIRouter()


# JWT настройки
SECRET_KEY = jwt_secret  # Замените на свой секретный ключ
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
token_auth_scheme = HTTPBearer()

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
    """
    Делает всю верификацию токенов через PyJWT
    """

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



# Routes
@test_router.get("/my_answers/")
def get_user_answers_count(token: str = Depends(token_auth_scheme)):
    """
    Возвращает ответы авторизованного пользователя
    """

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


@test_router.post("/create_test_manually/")# , response_model=QuestionBatchResponse)
async def create_test_manually(request: TestManualRequest):
    """
    Создает тест полностью вручную - все параметры передаются в request,
    а не генерируются нейросетью
    """
    
    db = SessionLocal()
    # add record to "Test" table
    new_test = Test(topic_id=1, #TODO: fetch topic from db
                    name=request.name,
                    difficulty=request.difficulty,
                    test_time=10, # temporary magic number
                    user_author_id=1) # TODO: figure out how to fetch user (i forgor)
    db.add(new_test)
    db.commit()
    db.refresh(new_test)

    for problem in request.problems:
        # add record to "Problem" table
        new_problem = Problem(question=problem.question, test_id=new_test.id)
        db.add(new_problem)
        db.commit()
        db.refresh(new_problem)
        for answer in problem.answers:
            # add record to "Answers" table
            new_answer = Answer(problem_id=new_problem.id, answer_content=answer.value, is_correct=answer.is_correct)
            db.add(new_answer)
            db.commit()
            db.refresh(new_answer)

    class TempModel(BaseModel):
        test_id: int = new_test.id
    return TempModel()


@test_router.post("/api/create_test_manually/")# , response_model=QuestionBatchResponse)
async def create_test_manually(request: TestManualRequest):
    """
    Костыль для взаимодействия с фронтом

    Создает тест полностью вручную - все параметры передаются в request,
    а не генерируются нейросетью.
    """

    db = SessionLocal()
    # add record to "Test" table
    new_test = Test(topic_id=1, #TODO: fetch topic from db
                    name=request.name,
                    difficulty=request.difficulty,
                    test_time=10, # temporary magic number
                    user_author_id=1) # TODO: figure out how to fetch user (i forgor)
    db.add(new_test)
    db.commit()
    db.refresh(new_test)

    for problem in request.problems:
        # add record to "Problem" table
        new_problem = Problem(question=problem.question, test_id=new_test.id)
        db.add(new_problem)
        db.commit()
        db.refresh(new_problem)
        for answer in problem.answers:
            # add record to "Answers" table
            new_answer = Answer(problem_id=new_problem.id, answer_content=answer.value, is_correct=answer.is_correct)
            db.add(new_answer)
            db.commit()
            db.refresh(new_answer)

    class TempModel(BaseModel):
        test_id: int = new_test.id
    return TempModel()


@test_router.post("/add_answer/", response_model=BaseResponse)
async def add_answer(answer: AnswerRequest, token: str = Depends(token_auth_scheme)):
    """
    Добавляет ответ к заданному вопросу
    """

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


@test_router.post("/api/generate_question/")# , response_model=QuestionBatchResponse)
async def create_test(request: TestRequest):
    """
    Я не помню зачем эта ручка здесь, но подозреваю что она была для проверки чего-то

    Но лучше лишний раз пусть повисит
    """

    # TODO: fetch test from DB by request.test_id
    class TempTest(BaseModel):
        testId: int = 1
    new_test = TempTest()
    return new_test


@test_router.get("/get_test/{test_id}")
async def get_test(test_id: int):
    """
    Получает тест из базы данных под номером {test_id}
    """

    db = SessionLocal()

    # Модель для ответа
    class TempAnswer(BaseModel):
        value: str
        is_correct: bool = False

    # Модель для вопроса
    class TempProblem(BaseModel):
        question: str
        answers: list[TempAnswer]

    # Модель для теста
    class TempTest(BaseModel):
        name: str
        problems: list[TempProblem]

    # Получаем тест из базы данных
    test = db.query(Test).filter(Test.id == test_id).first()
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")

    # Получаем все вопросы, связанные с этим тестом
    problems = db.query(Problem).filter(Problem.test_id == test_id).all()

    # Формируем список вопросов и ответов
    test_problems = []
    for problem in problems:
        # Получаем все ответы для текущего вопроса
        answers = db.query(Answer).filter(Answer.problem_id == problem.id).all()

        # Формируем список ответов
        problem_answers = [
            TempAnswer(value=answer.answer_content, is_correct=bool(answer.is_correct))
            for answer in answers
        ]

        # Добавляем вопрос и ответы в список
        test_problems.append(
            TempProblem(question=problem.question, answers=problem_answers)
        )

    # Формируем финальный объект теста
    test_data = TempTest(
        name=test.name,
        problems=test_problems
    )

    # Закрываем сессию базы данных
    db.close()

    return test_data

@test_router.get("/get_test_meta/{test_id}")
async def get_test_meta(test_id: int):
    """
    Получает метаданные о тесте: Предмет, Сложность, Тема и Промпт 
    (при отсутствии промпта возвращает "blank prompt")
    """

    db = SessionLocal()

    #Модель для метаданных
    class TestMeta(BaseModel):
        Subject: str
        Difficulty: str
        Topic: str
        Prompt: str

    # Получаем тест из базы данных
    test = db.query(Test).filter(Test.id == test_id).first()
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")

    test_topic = db.query(Topic).filter(Topic.id == test.topic_id).first()
    if not test_topic:
        test_topic = "Матрицы"
    
    test_subject = db.query(Subject).filter(Subject.id == test_topic.subject_id).first()
    if not test_subject:
        test_subject = "Математика"

    test_prompt = db.query(TestPrompt).filter(TestPrompt.test_id == test_id).first()
    if not test_prompt:
        test_prompt = "blank prompt" #TODO: Уточнить, что возвращать в случае если промпт не найден (а его у нас не будет сейчас - таблицу с промптами я создал только что)
    
    # Формируем финальный объект теста
    test_meta = TestMeta(
        Subject=test_subject.name,
        Difficulty=test.difficulty,
        Topic=test_topic.name,
        Prompt=test_prompt.prompt,
    )

    # Закрываем сессию базы данных
    db.close()

    return test_meta


@test_router.get("/get_user_tests/{user_id}")
async def get_user_tests(user_id: int):
    """
    Возвращает все тесты, которые были созданы пользователем под ID {user_id}
    """
    
    db = SessionLocal()

    #Модель для метаданных
    class TestMeta(BaseModel):
        Subject: str
        Difficulty: str
        Topic: str
        Prompt: str

    # Получаем тесты из базы данных
    tests_raw = db.query(Test).filter(Test.user_author_id == user_id)
    if not tests_raw:
        raise HTTPException(status_code=404, detail="No tests were found")

    # Модель для ответа
    class TempAnswer(BaseModel):
        value: str
        is_correct: bool = False

    # Модель для вопроса
    class TempProblem(BaseModel):
        question: str
        answers: list[TempAnswer]

    # Модель для теста
    class TempTest(BaseModel):
        name: str
        problems: list[TempProblem]
    
    tests = []
    for test in tests_raw:

        # Получаем все вопросы, связанные с этим тестом
        problems = db.query(Problem).filter(Problem.test_id == test.id).all()

        # Формируем список вопросов и ответов
        test_problems = []
        for problem in problems:
            # Получаем все ответы для текущего вопроса
            answers = db.query(Answer).filter(Answer.problem_id == problem.id).all()

            # Формируем список ответов
            problem_answers = [
                TempAnswer(value=answer.answer_content, is_correct=bool(answer.is_correct))
                for answer in answers
            ]

            # Добавляем вопрос и ответы в список
            test_problems.append(
                TempProblem(question=problem.question, answers=problem_answers)
            )

        # Формируем финальный объект теста
        test_data = TempTest(
            name=test.name,
            problems=test_problems
        )
        tests.append(test_data)

    # Закрываем сессию базы данных
    db.close()
    class Response(BaseModel):
        tests: list[TempTest]
    
    response = Response(tests = tests)
    return response


def generate_random():
    """
    Не ручка

    Генерирует случайное число от -10 до 10
    """

    random_value = random.randint(-10, 10)
    return str(random_value)


# Маршрут для генерации математических вопросов
# TODO : 1) ручка должна обращаться к task.db и искать по topic difficaltly все соответвующие задания 
# TODO : 2) собирать рандомные задания по 4 штуки из всех что нашла 
# TODO : 3) обрщаться к YandexGPT отдавая ей примеры и так же отдавая count также будет промт составить похожие задачи и решить их
# TODO : 4) то что будет присылаться от gpt мы будем схрянять правльный ответ сохранять а после брать ответ и менять цифры и сохранять в другую переменную 
# TODO : 5) правльный ответ не должен быть первым ( рандомно перемешать)

@test_router.post("/generate_math_quastion/")
async def generate_math_quastion(request: QuestionAutoGenerateRequest):
    try:
        conn = duckdb.connect("task.db")

        query = f"""
        SELECT latex_example
        FROM task
        WHERE topic = ? AND difficulty = ? 
        LIMIT ?
"""
        result = conn.execute( query,(request.topic, request.difficulty ))

        if not result:

            raise HTTPException(status_code=404, detail="No matching questions found")
        
        return result 
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))










# Маршрут для генерации математических вопросов
@test_router.post("/generate_question_from_text/")# , response_model=QuestionBatchResponse)
async def generate_question_from_text(request: QuestionFromTextRequest):
    # TODO: add Test Name field andput it in the DB
    """
    Генерирует по ТЕКСТУ математический вопрос и ответ с использованием OpenAI API.
    """

    db = SessionLocal()
    try:
        # TODO: rewrite prompt to generate a JSON
        new_batch = []
        # add record to "Test" table
        new_test = Test(topic_id=1, 
                        name="Autogenerated test", 
                        difficulty="medium", 
                        test_time=10, # temporary magic number
                        user_author_id=1) # TODO: figure out how to fetch user (i forgor)
        db.add(new_test)
        db.commit()
        db.refresh(new_test)

        new_question_batch = []
        for i in range(3): # Will be redundant - Ai decides how many questions are there needed
        #This prompt will only be used if a specific prompt wasn't found in the aformentioned json
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
                    "text": "Сгенерируй вопрос по тексту:" + \
                        request.text,
                    }
                ]
            }
            print("Response generating....")

            # Запрос к YandexGPT
            url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Api-Key AQVN3Gy8RXHx2J4ocgbGY47qr9UMigMSI2nwDJDF"
            }
            response = requests.post(url, headers=headers, json=prompt)

            # Извлечение ответа
            print(f"Answer {i} parsing....")
            content_raw = response.text
            content = json.loads(content_raw)["result"]["alternatives"][0]["message"]["text"]
            print(content)

            # Разделение на вопрос и ответ
            question = content.replace("Вопрос: ", "")

            # add record to "Problem" table
            new_problem = Problem(question=question.strip(), test_id=new_test.id)
            db.add(new_problem)
            db.commit()
            db.refresh(new_problem)
            new_question_batch.append(question.strip())

        print("returning")

        class TempQuestion(BaseModel):
            question: str

        class TempData(BaseModel):
            data: list[TempQuestion]

        data = TempData(
            data = [
                TempQuestion(question=question) for question in new_question_batch
            ]
        )
        
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации вопроса: {e}")
    

auth = VerifyToken()

token_auth_scheme = HTTPBearer()

async def fetchCurrentUser(sub, token):
    """
    Возвращает авторизованного пользователя
    """

    async with aiohttp.ClientSession(headers={
        "Authorization": f'Bearer {token}'
    }) as session:
        async with session.get(f'https://{AUTH0_DOMAIN}/api/v2/users/{sub}') as response:
            # Read the response text
            return await response.json()


# Эндпоинт для защищенных ресурсов
@test_router.get("/users/me")
async def read_users_me(auth_result = Security(auth.verify)):
    """
    Судя по всему тестовая ручка

    Эта ручка верифицирует авторизованного пользователя
    """

    print('Вызван защищённый авторизацией эндпоинт')
    print(json.dumps(auth_result, indent=4))
    return auth_result


@test_router.get("/private")
def private(token: str = Depends(token_auth_scheme)):
    """
    Тестовая ручка

    Для доступа к ней нужен валидный JWT токен
    A valid access token is required to access this route
    """

    result = jwt.decode(token.credentials, options={"verify_signature": False})

    return result


@test_router.post("/register/")
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
        # hashed_password = pwd_context.hash(password)
        # new_user = User(username=username, hashed_password=hashed_password, email=email)
        new_user = User(username=username, email=email, )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"username": new_user.username, "message": "Пользователь успешно зарегистрирован."}
    finally:
        db.close()
