from fastapi import FastAPI, Request, HTTPException, Form, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import aioboto3
import uvicorn
import asyncio
import json
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from openai import OpenAI
from src.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_SUMMARY


# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize templates
templates = Jinja2Templates(directory="src/templates")

# DynamoDB Configuration
DYNAMODB_ENDPOINT = os.getenv("DYNAMODB_ENDPOINT", "http://localhost:8000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "dummy")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "dummy")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

TABLE_NAME = "Users"

# --- Utility Functions ---
async def initialize_dynamodb_table():
    """
    Асинхронное создание таблицы в DynamoDB, если она не существует.
    """
    try:
        # Получаем клиент DynamoDB через вашу функцию
        dynamodb = await get_dynamodb_resource()

        # Проверяем, существует ли таблица
        existing_tables = await dynamodb.list_tables()
        if TABLE_NAME in existing_tables.get("TableNames", []):
            print(f"Таблица {TABLE_NAME} уже существует.")
            return

        # Создаём новую таблицу
        print(f"Создаём таблицу {TABLE_NAME}...")
        await dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[
                {"AttributeName": "user_id", "KeyType": "HASH"},  # Ключ (Primary Key)
            ],
            AttributeDefinitions=[
                {"AttributeName": "user_id", "AttributeType": "S"},  # Тип данных 'строка'
            ],
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        )
        print("Таблица успешно создана. Ожидание активации...")

        # Ждём, пока таблица станет активной
        waiter = dynamodb.get_waiter(TableName="Users")
        await waiter.wait(TableName=TABLE_NAME)
        print("Таблица активна и готова к работе!")

    except ClientError as e:
        # Обрабатываем ошибки, например, если таблица уже используется
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"Таблица {TABLE_NAME} уже существует.")
        else:
            print("Ошибка при создании таблицы:", e)
            raise

async def get_dynamodb_resource():
    session = aioboto3.Session()
    async with session.client(
        "dynamodb",
        endpoint_url=DYNAMODB_ENDPOINT,
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    ) as dynamodb:
        return dynamodb

def initialize_knowledge_base():
    global db
    VECTOR_DB_DIR = "src/vectorized_db"
    KNOWLEDGE_BASE_PATH = "src/knowledge_base.txt"

    if os.path.exists(VECTOR_DB_DIR):
        logger.info("Loading vectorized knowledge base from %s", VECTOR_DB_DIR)
        db = FAISS.load_local(VECTOR_DB_DIR, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            raise FileNotFoundError(f"Knowledge base file not found: {KNOWLEDGE_BASE_PATH}")

        logger.info("Vectorizing knowledge base...")
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            knowledge_data = f.read()

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1")])
        chunks = markdown_splitter.split_text(knowledge_data)
        db = FAISS.from_documents(chunks, OpenAIEmbeddings())
        db.save_local(VECTOR_DB_DIR)
        logger.info("Vectorization completed and saved to %s", VECTOR_DB_DIR)

@app.on_event("startup")
async def startup_event():
    try:
        # Инициализация таблицы в DynamoDB
        await initialize_dynamodb_table()
        logger.info("DynamoDB table initialized.")

        # Инициализация базы знаний (если необходимо)
        initialize_knowledge_base()
        logger.info("Knowledge base initialized.")
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        raise RuntimeError("Failed to initialize application.") from e

def summarize_dialog(summarized_dialog: str, last_interaction: dict) -> str:
    """
    Обновляем существующую саммаризацию новым вопросом и ответом.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
        {
            "role": "user",
            "content": (
                f"Текущий саммаризованный диалог:\n{summarized_dialog}\n\n"
                f"Новый вопрос и ответ:\n"
                f"Пользователь: {last_interaction.get('question', '')}\n"
                f"Консультант: {last_interaction.get('answer', '')}\n\n"
                f"Обнови саммаризацию, добавив новую информацию. Не превышай 5000 символов."
            ),
        }
    ]

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0
    )

    return completion.choices[0].message.content


async def get_user_data_from_db(user_id: str) -> dict:
    try:
        async with aioboto3.Session().client(
                "dynamodb",
                endpoint_url=DYNAMODB_ENDPOINT,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION,
        ) as dynamodb:
            response = await dynamodb.get_item(
                TableName=TABLE_NAME,
                Key={"user_id": {"S": user_id}}
            )
            user_data = response.get("Item")
            if not user_data:
                return None

            # Десериализация conversation_history (если оно есть)
            conversation_history = user_data.get("conversation_history", {}).get("S", "[]")
            try:
                conversation_history = json.loads(conversation_history)  # Преобразование из строки в список
            except json.JSONDecodeError:
                conversation_history = []  # Если не удается декодировать, используем пустой список

            return {
                "id": user_data.get("user_id", {}).get("S", ""),
                "name": user_data.get("name", {}).get("S", ""),
                "birthday": user_data.get("birthday", {}).get("S", ""),
                "user_info": user_data.get("user_info", {}).get("S", ""),
                "conversation_history": conversation_history
            }
    except ClientError as e:
        logger.error(f"Ошибка при запросе к таблице Users: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при запросе к базе данных.")
    except Exception as e:
        logger.error(f"Общая ошибка при взаимодействии с DynamoDB: {e}")
        raise HTTPException(status_code=500, detail="Общая ошибка при запросе к базе данных.")


# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/users", response_class=HTMLResponse)
async def get_users(request: Request):
    try:
        dynamodb = await get_dynamodb_resource()
        response = await dynamodb.scan(TableName=TABLE_NAME)
        users = response.get("Items", [])
        if not users:
            return RedirectResponse(url="/add_user", status_code=303)

        # Преобразование данных для отображения
        formatted_users = []
        for user in users:
            conversation_history = user.get("conversation_history", {}).get("S", "[]")
            try:
                conversation_history = json.loads(conversation_history)  # Преобразование из строки в список
            except json.JSONDecodeError:
                conversation_history = []  # Если JSON некорректный, используем пустой список

            formatted_users.append({
                "id": user.get("user_id", {}).get("S", ""),
                "name": user.get("name", {}).get("S", ""),
                "birthday": user.get("birthday", {}).get("S", ""),
                "user_info": user.get("user_info", {}).get("S", ""),
                "conversation_history": conversation_history  # Загружаем всю сохраненную историю диалога
            })

        return templates.TemplateResponse("users_table.html", {"request": request, "users": formatted_users})
    except Exception as e:
        logger.error(f"Ошибка при получении списка пользователей: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users.")


@app.get("/add_user", response_class=HTMLResponse)
async def add_user_form(request: Request):
    return templates.TemplateResponse("user_form.html", {"request": request})


@app.post("/add_user")
async def add_user(payload: dict):
    """
    Добавление или обновление пользователя в базе данных с сохранением существующих данных.
    """
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Поле user_id обязательно.")

    try:
        dynamodb = await get_dynamodb_resource()

        # Получаем текущие данные пользователя (если они существуют)
        response = await dynamodb.get_item(
            TableName=TABLE_NAME,
            Key={"user_id": {"S": user_id}}
        )
        current_data = response.get("Item", {})

        # Загружаем существующую историю сообщений, если есть
        existing_conversation_history = current_data.get("conversation_history", {}).get("S", "[]")
        try:
            existing_conversation_history = json.loads(existing_conversation_history)
        except json.JSONDecodeError:
            existing_conversation_history = []  # Если JSON некорректный, используем пустой список

        # Объединяем текущие данные с новыми значениями
        updated_data = {
            "user_id": {"S": user_id},
            "name": {"S": payload.get("name", current_data.get("name", {}).get("S", ""))},
            "birthday": {"S": payload.get("birthday", current_data.get("birthday", {}).get("S", ""))},
            "user_info": {"S": payload.get("user_info", current_data.get("user_info", {}).get("S", ""))},
            "conversation_history": {"S": json.dumps(existing_conversation_history)}  # Сохраняем историю
        }

        # Сохраняем обновленные данные
        await dynamodb.put_item(
            TableName=TABLE_NAME,
            Item=updated_data
        )

        return RedirectResponse(url="/users", status_code=303)
    except Exception as e:
        logger.error(f"Ошибка при добавлении/обновлении пользователя: {e}")
        raise HTTPException(status_code=500, detail="Failed to add/update user.")


def process_question_sync(user_prompt):
    # Отправляем запрос в OpenAI
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0
    )

    answer = completion.choices[0].message.content
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens

    metadata = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

    return answer, metadata

@app.get("/process_question", response_class=HTMLResponse)
async def interact_page(request: Request):
    return templates.TemplateResponse("interact.html", {"request": request, "answer": None, "metadata": None})


@app.post("/process_question")
async def process_question(request: Request, payload: dict = Body(...)):
    """
    Основной эндпоинт для обработки вопроса пользователя.
    Саммаризация вызывается всегда, независимо от флага use_conversation_history.
    """
    user_id = payload.get("user_id")
    question = payload.get("question")
    use_user_info = payload.get("use_user_info", False)
    use_knowledge_base = payload.get("use_knowledge_base", False)
    use_conversation_history = payload.get("use_conversation_history", False)

    if not user_id:
        raise HTTPException(status_code=400, detail="Поле user_id обязательно.")

    try:
        # 1. Получаем данные о пользователе
        user_data = await get_user_data_from_db(user_id)
        if not user_data:
            # Если пользователь не найден
            raise HTTPException(status_code=404, detail="Пользователь не найден.")

        # Извлекаем нашу структуру для диалога (messages, summarized_dialog, last_interaction)
        conversation_data = user_data.get("conversation_history", {})
        if not isinstance(conversation_data, dict):
            # Если вдруг сохранился список, подгоняем под новую схему
            conversation_data = {
                "messages": conversation_data if isinstance(conversation_data, list) else [],
                "summarized_dialog": "",
                "last_interaction": {}
            }

        messages = conversation_data.get("messages", [])
        summarized_dialog = conversation_data.get("summarized_dialog", "")
        last_interaction = conversation_data.get("last_interaction", {})

        # 2. Формируем user_prompt
        user_prompt = ""

        #   2.1. Используем информацию о пользователе
        if use_user_info:
            user_prompt += (
                "Информация о пользователе:\n"
                f"Имя: {user_data.get('name', '')}\n"
                f"Дата рождения: {user_data.get('birthday', '')}\n"
                f"Дополнительная информация: {user_data.get('user_info', '')}\n\n"
            )

        #   2.2. Используем историю диалога (если включено use_conversation_history)
        #        Но добавляем только summarized_dialog и last_interaction, а не все messages
        if use_conversation_history:
            if summarized_dialog:
                user_prompt += f"Сжатое изложение диалога:\n{summarized_dialog}\n\n"
            if last_interaction:
                user_prompt += "Последнее взаимодействие:\n"
                user_prompt += f"Вопрос: {last_interaction.get('question', '')}\n"
                user_prompt += f"Ответ: {last_interaction.get('answer', '')}\n\n"

        #   2.3. Используем базу знаний
        if use_knowledge_base:
            docs = db.similarity_search(question, k=3)
            snippet = "\n".join([
                f"\nОтрывок документа №{i+1}\n=====================\n{doc.page_content.strip()}\n"
                for i, doc in enumerate(docs)
            ])
            user_prompt += f"База знаний: {snippet}\n"

        #   2.4. Добавляем сам вопрос
        user_prompt += (
            f"\n\nВопрос: {question}. Ответь на вопрос на основе предоставленной информации и собственных "
            f"Ответь на вопрос на основе предоставленной информации и собственных знаний. Можно отвечать только на вопросы, связанные с жилищным законодательством РФ."
        )

        logger.info(f"Сформированный user_prompt: {user_prompt}")

        # 3. Запрашиваем у GPT ответ
        answer, metadata = await asyncio.to_thread(process_question_sync, user_prompt)

        # 4. Обновляем историю сообщений (messages)
        new_entry_user = {"role": "user", "message": question, "timestamp": int(asyncio.get_event_loop().time())}
        new_entry_model = {"role": "model", "message": answer, "timestamp": int(asyncio.get_event_loop().time())}
        messages.append(new_entry_user)
        messages.append(new_entry_model)
        # Храним только последние 10 сообщений
        messages = messages[-10:]
        conversation_data["messages"] = messages

        # 5. Формируем новый last_interaction
        new_last_interaction = {"question": question, "answer": answer}
        conversation_data["last_interaction"] = new_last_interaction

        # 6. Обновляем summarized_dialog (саммаризация вызывается всегда)
        updated_summarized = await asyncio.to_thread(summarize_dialog, summarized_dialog, new_last_interaction)
        conversation_data["summarized_dialog"] = updated_summarized

        # 7. Сохраняем обновлённые данные
        user_data["conversation_history"] = conversation_data

        # Обновляем запись в DynamoDB
        async with aioboto3.Session().client(
            "dynamodb",
            endpoint_url=DYNAMODB_ENDPOINT,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        ) as dynamodb:
            await dynamodb.put_item(
                TableName=TABLE_NAME,
                Item={
                    "user_id": {"S": user_id},
                    "name": {"S": user_data.get("name", "")},
                    "birthday": {"S": user_data.get("birthday", "")},
                    "user_info": {"S": user_data.get("user_info", "")},
                    "conversation_history": {"S": json.dumps(conversation_data)}
                }
            )

        # 8. Возвращаем ответ
        return JSONResponse(content={"answer": answer, "metadata": metadata})

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {e}")
        return JSONResponse(content={
            "error": "Ошибка: невозможно обработать запрос"
        }, status_code=500)



if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8080, reload=True)
