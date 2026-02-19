import asyncio
import os
import json
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, WebAppInfo, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardBuilder

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# === ПРОВЕРКА ТОКЕНОВ ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
# ЗАМЕНИ НА СВОЮ ССЫЛКУ GITHUB PAGES
WEB_APP_URL = "https://dmitriilikhosherst24.github.io/sretensk-rag-telegram-bot/" 

if not BOT_TOKEN or not HF_TOKEN:
    raise ValueError("❌ Нет токенов в переменных окружения!")

# === НАСТРОЙКА AI ===
# Модель для поиска (должна совпадать с той, которой создавалась база)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Загружаем базу
db = FAISS.load_local("sretensk_db", embeddings, allow_dangerous_deserialization=True)

# LLM (Мозг)
endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.3, # Баланс: не робот, но и не сказочник
    max_new_tokens=2048,
)
llm = ChatHuggingFace(llm=endpoint)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# === ПРОМПТ ===
SYSTEM_PROMPT = """
Ты — методист-консультант Сретенской духовной академии.
Твоя задача — помогать студентам, основываясь на нормативных актах.

ИНСТРУКЦИЯ:
1. Внимательно изучи предоставленный КОНТЕКСТ.
2. Ответь на вопрос пользователя, используя факты из контекста.
3. Если в контексте есть частичная информация — используй её. Не говори "я не знаю", если в тексте есть хоть что-то полезное.
4. Если информации совсем нет — предложи обратиться в деканат.
5. Ссылайся на названия документов, если они указаны в тексте.
"""

async def get_answer(question: str):
    try:
        # Ищем 10 фрагментов (было 6, стало больше, чтобы не терять инфо)
        docs = db.similarity_search(question, k=10)
        
        # Собираем контекст
        context_text = ""
        for d in docs:
            # Отсеиваем слишком короткий мусор
            if len(d.page_content) > 40:
                context_text += f"\n--- ИЗ ДОКУМЕНТА: {d.metadata.get('source', 'Неизвестно')} ---\n{d.page_content}\n"

        if not context_text:
            return "К сожалению, в базе знаний не нашлось подходящих документов. Попробуйте сформулировать иначе."

        # Запрос к нейросети
        response = await llm.ainvoke([
            ("system", SYSTEM_PROMPT),
            ("human", f"КОНТЕКСТ:\n{context_text}\n\nВОПРОС СТУДЕНТА: {question}")
        ])
        return response.content

    except Exception as e:
        print(f"Ошибка AI: {e}")
        return "Произошла техническая ошибка при обработке запроса."

# === ОБРАБОТЧИКИ (HANDLERS) ===

@dp.message(Command("start"))
async def start_handler(message: Message):
    # Создаем клавиатуру с кнопкой WebApp
    kb = ReplyKeyboardBuilder()
    kb.button(text="📱 Открыть Вопросы", web_app=WebAppInfo(url=WEB_APP_URL))
    
    await message.answer(
        "👋 Привет! Я помощник по документам Академии.\n"
        "Нажми кнопку ниже, чтобы выбрать тему или задать вопрос.",
        reply_markup=kb.as_markup(resize_keyboard=True)
    )

# ЛОВИМ ДАННЫЕ ИЗ МИНИ-АППА
@dp.message(F.web_app_data)
async def web_app_data_handler(message: Message):
    print(f"📥 Пришли данные из WebApp: {message.web_app_data.data}") # Лог для отладки
    
    try:
        data = json.loads(message.web_app_data.data)
        question = data.get("question")
        
        if question:
            # Пишем юзеру, что процесс пошел
            wait_msg = await message.answer(f"🔍 Ищу: <b>{question}</b>...", parse_mode="HTML")
            
            # Генерируем ответ
            answer = await get_answer(question)
            
            # Удаляем "Ищу..." и пишем ответ
            await wait_msg.delete()
            await message.answer(answer)
            
    except Exception as e:
        await message.answer(f"Ошибка чтения данных: {e}")

# ОБЫЧНЫЙ ТЕКСТ
@dp.message()
async def text_handler(message: Message):
    if message.text:
        wait_msg = await message.answer("🔍 Читаю документы...")
        answer = await get_answer(message.text)
        await wait_msg.delete()
        await message.answer(answer)

async def main():
    print("🚀 Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
