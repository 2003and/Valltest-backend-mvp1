from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel

Base = declarative_base()

# Tables
class Test(Base):
    __tablename__ = "test"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    difficulty = Column(String)
    topic_id = Column(Integer, ForeignKey("topic.id", ondelete="CASCADE"))
    # creation_date = Column(String)
    test_time = Column(Integer)
    user_author_id = Column(String, ForeignKey("user.id", ondelete="CASCADE"))


class Problem(Base):
    __tablename__ = "problem"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String) # , unique=True, index=True
    test_id = Column(Integer, ForeignKey("topic.id", ondelete="CASCADE"))


class Answer(Base):
    __tablename__ = "answer"
    id = Column(Integer, primary_key=True, index=True)
    problem_id = Column(Integer, ForeignKey("problem.id", ondelete="CASCADE"))
    answer_content = Column(String)
    is_correct = Column(Integer)


class UserAnswers(Base):
    __tablename__ = "user_answers"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"))
    problem_id = Column(Integer, ForeignKey("problem.id", ondelete="CASCADE"))
    answer_num = Column(Integer)


class Subject(Base):
    __tablename__ = "subject"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)


class Topic(Base):
    __tablename__ = "topic"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    subject_id = Column(Integer, ForeignKey("subject.id", ondelete="CASCADE"))


class TestPrompt(Base):
    __tablename__ = "test_prompt"
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(Integer, ForeignKey("test.id", ondelete="CASCADE"))
    prompt = Column(String)

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    # hashed_password = Column(String)
    email = Column(String, unique=True)
    # role = Column(String) #TODO: add roles and permissions functionality

# Validators
class AnswerModel(BaseModel):
    value: str
    is_correct: bool

class ProblemModel(BaseModel):
    question: str
    answers: list[AnswerModel]

class TestManualRequest(BaseModel):
    name: str = "untitled test"
    topic: str = "integral"
    subject: str = "math"
    difficulty: str = "easy" # Уровень сложности: easy, medium, hard
    amount: int = 10
    problems: list[ProblemModel]
    #tags

class TokenData(BaseModel):
    username: str

class TestRequest(BaseModel):
    id: int = 1

# Модель для запроса и ответа
class QuestionAutoGenerateRequest(BaseModel):
    topic: str = "integral"
    subject: str = "math"
    difficulty: str = "easy" # Уровень сложности: easy, medium, hard
    amount: int = 10
    name: str = "untitled test"

class QuestionFromTextRequest(BaseModel):
    text: str = "[your prompt here]"

class BaseResponse(BaseModel):
    response: str
    
class AnswerRequest(BaseModel):
    problem_num: int
    answer_num: int
    correct_answer_num: int