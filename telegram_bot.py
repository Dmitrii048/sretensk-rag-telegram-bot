import asyncio
import os
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, WebAppInfo

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# === ПЕРЕМЕННЫЕ ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN  = os.getenv("HF_TOKEN")

if not BOT_TOKEN or not HF_TOKEN:
    raise ValueError("❌ Не указаны BOT_TOKEN или HF_TOKEN в переменных окружения!")

# База и модель (загружаются один раз)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("sretensk_db", embeddings, allow_dangerous_deserialization=True)

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",  # 7B — быстрее и легче для хостинга
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.3,
    max_new_tokens=2048,
)
llm = ChatHuggingFace(llm=endpoint)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# --- СИСТЕМНЫЙ ПРОМПТ ---
SYSTEM_PROMPT = """
Ты — официальный методист-юрист Сретенской духовной академии.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Обязательно ссылайся на названия документов и номера фрагментов.
Если прямого ответа нет, но можно логично вывести — сделай вывод.
Если совсем ничего подходящего — скажи ровно: 'В документах нет информации по этому вопросу.'
Стиль: официальный, понятный, вежливый.
"""

async def generate_answer(question: str) -> str:
    try:
        docs = db.similarity_search(question, k=8)
        docs = [d for d in docs if len(d.page_content.strip()) > 50]

        if not docs:
            return "В документах нет информации по этому вопросу. Попробуйте переформулировать или уточнить."

        context = "\n\n".join([
            f"--- Фрагмент из {d.metadata.get('source', 'документа')} ---\n{d.page_content.strip()}"
            for d in docs
        ])

        response = await llm.ainvoke([
            ("system", SYSTEM_PROMPT),
            ("human", f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}")
        ])

        return response.content
    except Exception as e:
        return f"Ошибка: {str(e)[:200]}"

@dp.message(Command("start"))
async def start_handler(message: Message):
    markup = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(
                text="📱 Открыть помощника",
                web_app=WebAppInfo(url="https://твой-ник.github.io/sretensk-rag-telegram-bot/miniapp/")
            )]
        ],
        resize_keyboard=True,
        one_time_keyboard=False
    )

    await message.answer(
        "👋 Добро пожаловать! Я — интеллектуальный ассистент Сретенской духовной академии.\n\n"
        "Можешь писать вопросы сюда или нажать кнопку ниже, чтобы открыть удобный чат.",
        reply_markup=markup
    )

@dp.message()
async def text_handler(message: Message):
    if message.text:
        await message.answer("🔍 Ищу в документах...")
        answer = await generate_answer(message.text)
        await message.answer(answer)

async def main():
    print("Бот запущен!")
    await dp.start_polling(bot, drop_pending_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
