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
# Вставь ссылку на GitHub Pages
WEB_APP_URL = "https://dmitriilikhosherst24.github.io/sretensk-rag-telegram-bot/" 

if not BOT_TOKEN or not HF_TOKEN:
    raise ValueError("❌ Проверь токены!")

# === AI ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("sretensk_db", embeddings, allow_dangerous_deserialization=True)

# СТРОГИЕ НАСТРОЙКИ
endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.1,  # <--- ОЧЕНЬ ВАЖНО: Убираем фантазии почти в ноль
    max_new_tokens=2048,
)
llm = ChatHuggingFace(llm=endpoint)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# === СИСТЕМНЫЙ ПРОМПТ (АНТИ-ГАЛЛЮЦИНАЦИИ) ===
SYSTEM_PROMPT = """
Ты — строгий методист-юрист Сретенской духовной академии.
Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленного ниже КОНТЕКСТА.

ПРАВИЛА:
1. Если в контексте нет информации о ВСОКО, олимпиадах или чем-то еще — так и скажи: "В доступных мне документах (Устав, Положения) нет информации об этом."
2. ЗАПРЕЩЕНО придумывать факты или брать их из общей эрудиции.
3. Всегда указывай название документа, если оно есть в контексте.
4. В конце ответа предложи 2-3 уточняющих вопроса по теме найденного документа.
"""

async def generate_smart_answer(question: str):
    try:
        # Ищем документы
        docs = db.similarity_search(question, k=6)
        # Фильтр мусора
        docs = [d for d in docs if len(d.page_content.strip()) > 30]

        if not docs:
            return "⚠️ В базе знаний Академии не найдено документов, соответствующих вашему запросу. Попробуйте переформулировать вопрос (например, используйте более официальные термины)."

        # Собираем контекст
        context_text = "\n\n".join([f"📄 Источник: {d.metadata.get('source', 'Документ')}\n{d.page_content}" for d in docs])

        # Формируем запрос
        response = await llm.ainvoke([
            ("system", SYSTEM_PROMPT),
            ("human", f"КОНТЕКСТ:\n{context_text}\n\nВОПРОС: {question}")
        ])
        return response.content
    except Exception as e:
        return f"Произошла техническая ошибка: {str(e)[:100]}"

# === ХЕНДЛЕРЫ ===

@dp.message(Command("start"))
async def start_cmd(message: Message):
    # КЛАВИАТУРА ПОД СТРОКОЙ ВВОДА (Самый надежный способ)
    kb = ReplyKeyboardBuilder()
    kb.button(text="🎓 Открыть помощника", web_app=WebAppInfo(url=WEB_APP_URL))
    
    await message.answer(
        "👋 Здравствуйте! Я правовой ассистент СДА.\n"
        "Нажмите кнопку внизу, чтобы открыть удобный интерфейс поиска.",
        reply_markup=kb.as_markup(resize_keyboard=True)
    )

# ЛОВИМ ДАННЫЕ ИЗ МИНИ-АППА
@dp.message(F.content_type == ContentType.WEB_APP_DATA)
async def web_app_handler(message: Message):
    # 1. Читаем данные
    data = json.loads(message.web_app_data.data)
    question = data.get("question", "")
    
    if not question:
        return

    # 2. Отвечаем пользователю, что приняли запрос
    status_msg = await message.answer(f"📥 <b>Запрос принят:</b> {question}\n⏳ Ищу информацию...", parse_mode="HTML")
    
    # 3. Генерируем ответ
    answer = await generate_smart_answer(question)
    
    # 4. Удаляем сообщение "Ищу..." и пишем ответ (или просто пишем новое)
    await status_msg.delete()
    await message.answer(answer)

# ОБРАБОТКА ОБЫЧНОГО ТЕКСТА
@dp.message()
async def text_handler(message: Message):
    if not message.text: return
    msg = await message.answer("🔍 Анализирую документы...")
    answer = await generate_smart_answer(message.text)
    await msg.delete()
    await message.answer(answer)

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
