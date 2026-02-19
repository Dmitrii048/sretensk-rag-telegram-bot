import asyncio
import os
import json
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, WebAppInfo, ContentType
from aiogram.utils.keyboard import ReplyKeyboardBuilder

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# === НАСТРОЙКИ ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN  = os.getenv("HF_TOKEN")
# Вставь сюда свою ссылку на GitHub Pages, где лежит index.html
WEB_APP_URL = "https://твоя-ссылка-на-github-pages/" 

if not BOT_TOKEN or not HF_TOKEN:
    raise ValueError("❌ Не указаны токены!")

# === ИНИЦИАЛИЗАЦИЯ AI ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("sretensk_db", embeddings, allow_dangerous_deserialization=True)

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.4, # Чуть выше креативность, чтобы он предлагал вопросы
    max_new_tokens=2048,
)
llm = ChatHuggingFace(llm=endpoint)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# === УМНЫЙ ПРОМПТ ===
SYSTEM_PROMPT = """
Ты — ведущий методист-консультант Сретенской духовной академии.
Твоя задача — максимально подробно отвечать на вопросы, используя контекст.

ИНСТРУКЦИЯ:
1. Используй найденный контекст для ответа.
2. Если точного ответа нет — не останавливайся! Попробуй проанализировать контекст и дать "Вероятный порядок действий" или "Рекомендацию", но пометь это как совет.
3. Тон: уважительный, академический, заботливый.
4. В КОНЦЕ ОТВЕТА ОБЯЗАТЕЛЬНО:
   На основе темы вопроса сгенерируй 3 коротких дополнительных вопроса, которые могут заинтересовать студента.
   Оформи их списком после фразы "📌 Возможные уточнения:".
"""

async def generate_smart_answer(question: str):
    try:
        # Углубленный поиск: берем больше фрагментов (k=12)
        docs = db.similarity_search(question, k=12)
        # Фильтруем совсем короткий мусор
        docs = [d for d in docs if len(d.page_content.strip()) > 30]

        if not docs:
            return "К сожалению, в базе знаний нет точных инструкций по этому запросу. Попробуйте переформулировать вопрос, используя официальные термины (например, 'отчисление', 'академический отпуск')."

        context = "\n\n".join([f"--- Документ: {d.metadata.get('source', 'Нормативный акт')} ---\n{d.page_content}" for d in docs])

        # Запрос к нейросети
        response = await llm.ainvoke([
            ("system", SYSTEM_PROMPT),
            ("human", f"КОНТЕКСТ:\n{context}\n\nВОПРОС СТУДЕНТА: {question}")
        ])
        return response.content
    except Exception as e:
        return f"Произошла техническая ошибка: {str(e)[:100]}"

# === ХЕНДЛЕРЫ ===

@dp.message(Command("start"))
async def start_cmd(message: Message):
    # Кнопка для открытия Mini App
    kb = ReplyKeyboardBuilder()
    kb.button(text="🎓 Задать вопрос (Mini App)", web_app=WebAppInfo(url=WEB_APP_URL))
    
    await message.answer(
        "👋 Здравствуйте! Я интеллектуальный помощник Академии.\n"
        "Вы можете писать вопросы прямо здесь или использовать удобное приложение с кнопкой ниже.",
        reply_markup=kb.as_markup(resize_keyboard=True)
    )

# 1. Обработка данных ИЗ Mini App (когда нажали "Отправить" в веб-приложении)
@dp.message(F.content_type == ContentType.WEB_APP_DATA)
async def web_app_handler(message: Message):
    data = json.loads(message.web_app_data.data)
    question = data.get("question", "")
    
    if not question:
        return

    # Отвечаем в чат, так как Mini App закроется
    await message.answer(f"📥 Получен вопрос из приложения:\n<b>{question}</b>", parse_mode="HTML")
    await message.answer("⏳ Анализирую нормативные акты...")
    
    answer = await generate_smart_answer(question)
    await message.answer(answer)

# 2. Обработка обычного текста в чате
@dp.message()
async def text_handler(message: Message):
    if not message.text: return
    await message.answer("🔍 Ищу информацию...")
    answer = await generate_smart_answer(message.text)
    await message.answer(answer)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
