import asyncio
import os
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, WebAppInfo
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# --- 1. ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN  = os.getenv("HF_TOKEN")

if not BOT_TOKEN or not HF_TOKEN:
    raise ValueError("❌ Не указаны BOT_TOKEN или HF_TOKEN в переменных окружения!")

# --- 2. НАСТРОЙКА БАЗЫ И МОДЕЛИ ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Убедись, что папка sretensk_db лежит рядом с файлом бота
db = FAISS.load_local("sretensk_db", embeddings, allow_dangerous_deserialization=True)

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.3, # Чуть повысил, чтобы вопросы были живее
    max_new_tokens=2048,
)

llm = ChatHuggingFace(llm=endpoint)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# --- 3. НОВЫЙ ПРОМПТ (МОЗГИ) ---
SYSTEM_INSTRUCTION = """
Ты — проактивный методист-консультант Сретенской духовной академии.
Твоя задача: помогать студентам разбираться в документах.

ИНСТРУКЦИЯ:
1. Изучи предоставленный КОНТЕКСТ и ВОПРОС.
2. Если в контексте есть прямой ответ — дай его, сославшись на документ.
3. Если информации недостаточно или вопрос слишком общий (например, "Как отчислиться?"):
   - НЕ придумывай ответ.
   - ЗАДАЙ уточняющий вопрос пользователю (например: "Уточните, отчисление по собственному желанию или за неуспеваемость?").
4. В конце ответа всегда будь вежлив и предложи помощь с деталями.

Отвечай на русском языке.
"""

# --- 4. ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ (Генерация ответа) ---
async def generate_response(user_text: str):
    """
    Эта функция ищет документы и спрашивает нейросеть.
    Используется и для обычного чата, и для Mini App.
    """
    try:
        # Поиск в базе
        docs = db.similarity_search(user_text, k=4)
        
        # Фильтруем слишком короткие куски
        docs = [d for d in docs if len(d.page_content.strip()) > 20]

        if not docs:
            return "В моих базах данных нет точной информации по этому вопросу. Попробуйте переформулировать или обратитесь в деканат."

        # Собираем контекст
        context = "\n\n".join([
            f"--- Источник: {d.metadata.get('source', 'Документ')} ---\n{d.page_content.strip()}"
            for d in docs
        ])

        # Спрашиваем LLM
        response = await llm.ainvoke([
            ("system", SYSTEM_INSTRUCTION),
            ("human", f"КОНТЕКСТ:\n{context}\n\nВОПРОС СТУДЕНТА: {user_text}")
        ])
        
        return response.content

    except Exception as e:
        return f"Произошла ошибка при генерации ответа: {str(e)[:200]}"

# --- 5. ХЕНДЛЕРЫ (ОБРАБОТЧИКИ) ---

@dp.message(Command("start"))
async def start(message: Message):
    # !!!!!!! ВСТАВЬ СЮДА ССЫЛКУ НА СВОЙ GITHUB PAGES !!!!!!!
    # Пример: https://dmitry.github.io/my-bot-repo/
    web_app_url = "https://ТВОЙ_НИК.github.io/ТВОЙ_РЕПОЗИТОРИЙ/" 
    
    markup = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📱 Открыть помощника", web_app=WebAppInfo(url=web_app_url))]
        ],
        resize_keyboard=True
    )
    
    await message.answer(
        "👋 Привет! Я — интеллектуальный ассистент СДА.\n\n"
        "Вы можете писать вопросы прямо сюда или нажать кнопку ниже, "
        "чтобы открыть красивый интерфейс.",
        reply_markup=markup
    )

# Обработчик данных из Mini App (Web App)
@dp.message(F.web_app_data)
async def web_app_handler(message: Message):
    user_question = message.web_app_data.data # Получаем текст из Mini App
    
    # Визуальное подтверждение
    await message.answer(f"📥 Получен вопрос из приложения:\n_{user_question}_", parse_mode="Markdown")
    await message.answer("🔍 Анализирую документы...")
    
    # Генерируем ответ
    answer = await generate_response(user_question)
    await message.answer(answer)

# Обработчик обычных текстовых сообщений
@dp.message()
async def handle_text_question(message: Message):
    if not message.text: return
    
    await message.answer("🔍 Ищу информацию...")
    answer = await generate_response(message.text)
    await message.answer(answer)

# --- 6. ЗАПУСК ---
async def main():
    print("Бот запущен!")
    await dp.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
