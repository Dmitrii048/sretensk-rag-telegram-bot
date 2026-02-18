import asyncio
import os
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# Токены берутся из переменных окружения Railway
BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN  = os.getenv("HF_TOKEN")

if not BOT_TOKEN or not HF_TOKEN:
    raise ValueError("❌ Не указаны BOT_TOKEN или HF_TOKEN в переменных окружения!")

# База и модель
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("sretensk_db", embeddings, allow_dangerous_deserialization=True)

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.15,
    max_new_tokens=2048,
)
llm = ChatHuggingFace(llm=endpoint)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "👋 Привет! Я — интеллектуальный ассистент Сретенской духовной академии.\n\n"
        "Задавай вопросы по документам, отчислению, ВСОКО, аспирантуре и т.д."
    )

@dp.message()
async def handle_question(message: Message):
    await message.answer("🔍 Ищу в документах...")

    try:
        docs = db.similarity_search(message.text, k=10)
        docs = [d for d in docs if len(d.page_content.strip()) > 40]

        if not docs:
            await message.answer("В документах нет информации по этому вопросу.")
            return

        context = "\n\n".join([
            f"--- {d.metadata.get('source', 'документ')} ---\n{d.page_content.strip()}"
            for d in docs
        ])

        response = await llm.ainvoke([
            ("system", "Ты — официальный методист-юрист Сретенской духовной академии. Отвечай только по контексту. Ссылайся на источники."),
            ("human", f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {message.text}")
        ])

        await message.answer(response.content)

    except Exception as e:
        await message.answer(f"Ошибка: {str(e)[:200]}")

async def main():
    print("Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
