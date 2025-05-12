# from __future__ import annotations
# from yandex_cloud_ml_sdk import YCloudML
# from backend_secrets import CATALOG_ID, YANDEXGPT_SECRET_KEY
# messages = [
#     {
#         "role": "system",
#         "text": "Найди ошибки в тексте и исправь их",
#     },
#     {
#         "role": "user",
#         "text": """Ламинат подойдет для укладке на кухне или в детской 
# комнате – он не боиться влаги и механических повреждений благодаря 
# защитному слою из облицованных меламиновых пленок толщиной 0,2 мм и 
# обработанным воском замкам.""",
#     },
# ]


# def main():
#     sdk = YCloudML(
#         folder_id=CATALOG_ID,
#         auth=YANDEXGPT_SECRET_KEY,
#     )

#     result = (
#         sdk.models.completions("yandexgpt").configure(temperature=0.5).run(messages)
#     )

#     for alternative in result:
#         print(alternative)


# if __name__ == "__main__":
#     main()



from __future__ import annotations
import duckdb
from yandex_cloud_ml_sdk import YCloudML
from fastapi import HTTPException
from typing import List, Dict
from backend_secrets import CATALOG_ID, YANDEXGPT_SECRET_KEY
from src.tests.routes import QuestionAutoGenerateRequest, test_router

def generate_questions_with_ai(topic: str, difficulty: str, amount: int, db_questions: List[Dict]) -> str:
    messages = [
        {
            "role": "system",
            "text": f"""Ты - эксперт по составлению математических задач. 
            На основе предоставленных примеров составь новые, аналогичные задачи.
            Сохраняй стиль, сложность и формат оригинальных задач.
            Решения к задачам должны быть полными и точными.""",
        },
        {
            "role": "user",
            "text": f"""Составь {amount} похожих математических задач по теме "{topic}" 
            сложности "{difficulty}", аналогичных следующим примерам:
            
            Примеры задач:
            {db_questions}
            
            Требования:
            1. Формат каждой задачи должен быть как в примерах
            2. Укажи полное решение для каждой задачи
            3. Сложность должна соответствовать указанной
            4. Задачи должны быть разными, но на ту же тему""",
        }
    ]

    sdk = YCloudML(
        folder_id=CATALOG_ID,
        auth=YANDEXGPT_SECRET_KEY,
    )

    try:
        # Немного повышаем температуру для разнообразия 
        result = sdk.models.completions("yandexgpt") \
            .configure(temperature=0.7)\
            .run(messages)
        
        generated_questions = []
        for alternative in result:
            generated_questions.append(alternative.text)
        
        return "\n\n".join(generated_questions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


# Пример использования в вашей ручке
@test_router.post("/generate_math_question/")
async def generate_math_question(request: QuestionAutoGenerateRequest):
    print("Received:", request.topic, request.difficulty, request.amount)
    try:
        with duckdb.connect("tasks.db") as conn:
            query = f"""
            SELECT text, latex_example
            FROM tasks.tasks
            WHERE topic = '{request.topic}' AND difficulty = '{request.difficulty}'
            """
            
            result = conn.sql(query).fetchall()

            if not result:
                raise HTTPException(status_code=404, detail="No matching questions found")

            db_questions = [
                {"text": item[0], "latex_example": item[1]}
                for item in result
            ]
            
            # Генерируем новые вопросы с помощью AI
            ai_generated = generate_questions_with_ai(
                request.topic,
                request.difficulty,
                request.amount,
                db_questions
            )
            
            return {
                "original_questions": db_questions,
                "ai_generated_questions": ai_generated
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))