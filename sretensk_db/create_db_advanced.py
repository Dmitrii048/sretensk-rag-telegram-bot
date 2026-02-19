import os
import requests
from bs4 import BeautifulSoup 
from urllib.parse import urljoin, urlparse
# Импорты LangChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# ================= НАСТРОЙКИ =================
SOURCE_FOLDER = "data1" # Твоя папка с документами
DB_PATH = "sretensk_db" # Папка, куда сохранится база (та же, что в боте!)
# Ссылки, с которых начнем обход сайта + все важные страницы и документы
START_URLS = [
    # Главная и общие
    "https://sdamp.ru/",
    "https://sdamp.ru/contacts/",

    # Об Академии
    "https://sdamp.ru/about/",
    "https://sdamp.ru/about/akademia-segodnya/",
    "https://sdamp.ru/about/sotrudnichestvo/",
    "https://sdamp.ru/about/shkola-abiturienta/",

    # Структура Академии
    "https://sdamp.ru/structure/",
    "https://sdamp.ru/structure/scheme/",
    "https://sdamp.ru/structure/rector/",
    "https://sdamp.ru/structure/administratsiya-sotrudniki/",
    "https://sdamp.ru/structure/uchenyy-sovet/",
    "https://sdamp.ru/structure/cathedra/",
    "https://sdamp.ru/structure/cathedra/theology/",
    "https://sdamp.ru/structure/cathedra/pastyr/",
    "https://sdamp.ru/structure/cathedra/history/",
    "https://sdamp.ru/structure/cathedra/yazyk/",
    "https://sdamp.ru/structure/cathedra/prakt-gum/",
    "https://sdamp.ru/structure/other-structures/",
    "https://sdamp.ru/life/studencheskiy-sovet/",

    # Сведения об образовательной организации
    "https://sdamp.ru/sveden/",
    "https://sdamp.ru/sveden/common/",
    "https://sdamp.ru/sveden/struct/",
    "https://sdamp.ru/sveden/document/",
    "https://sdamp.ru/sveden/education/",
    "https://sdamp.ru/sveden/managers/",
    "https://sdamp.ru/sveden/employees/",
    "https://sdamp.ru/sveden/objects/",
    "https://sdamp.ru/sveden/paid_edu/",
    "https://sdamp.ru/sveden/budget/",
    "https://sdamp.ru/sveden/vacant/",
    "https://sdamp.ru/sveden/grants/",
    "https://sdamp.ru/sveden/inter/",
    "https://sdamp.ru/sveden/catering/",
    "https://sdamp.ru/sveden/eduStandarts/",
    "https://sdamp.ru/sveden/quality/",
    "https://sdamp.ru/sveden/eios/",

    # Абитуриенту и образование
    "https://sdamp.ru/abitur/",
    "https://sdamp.ru/abitur/bachelor/",
    "https://sdamp.ru/abitur/magistr/",
    "https://sdamp.ru/abitur/aspirant/",
    "https://sdamp.ru/abitur/faq/",
    "https://sdamp.ru/obrazovanie/bakalavriat/",
    "https://sdamp.ru/obrazovanie/magistratura/",
    "https://sdamp.ru/obrazovanie/aspirantura/",
    "https://sdamp.ru/obrazovanie/prepodavateli/",
    "https://sdamp.ru/obrazovanie/raspisanie-zanyatiy/Расписание%20Бак%20и%20Маг%202026.pdf",

    # Наука
    "https://sdamp.ru/nauka/biblioteka/",
    "https://sdamp.ru/nauka/conferences/",
    "https://sdamp.ru/nauka/sretenskiy-sbornik/",
    "https://sdamp.ru/nauka/sretenskoe-slovo/",
    "https://sdamp.ru/nauka/zhurnal-diakrisis/",
    "https://sdamp.ru/nauka/dokt-sovet/",
    "https://sdamp.ru/nauka/stud-nauka/",

    # Все важные PDF-документы (добавлены напрямую, т.к. краулер их пропускает)
    "https://sdamp.ru/upload/Перечень_документов_на_конкурс_на_замещение_должностей_педагогических_2025.pdf",
    "https://sdamp.ru/upload/Квалификационные_требования_к_должностям_педагогических_работников_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/ris/Informaciya_o_predostavlenii_platnyx_obrazovatelynyx_uslug_tolyko_za_schet_sredstv_Akademii.pdf",
    "https://sdamp.ru/vikon/sveden/files/aij/Dogovor_ob_obrazovanii_511_Teologiya_byudghet_graghdane_RF_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/viy/Dogovor_ob_obrazovanii_511_Teologiya_byudghet_inostrannye_graghdane_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/aza/Dogovor_ob_obrazovanii_511_Teologiya_platnoe_obuchenie_inostrannye_graghdane_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/zik/Dogovor_ob_obrazovanii_000_Podgotovka_(aspirantura)_graghdane_RF_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/vig/Dogovor_ob_obrazovanii_000_Podgotovka_(aspirantura)_inostrannye_graghdane_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/viy/Dogovor_ob_obrazovanii_480301_Teologiya_byudghet_graghdane_RF_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/eiu/Dogovor_ob_obrazovanii_480301_Teologiya_byudghet_inostrannye_graghdane_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/vig/Dogovor_ob_obrazovanii_480301_Teologiya_platnoe_obuchenie_graghdane_RF_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/aiq/Dogovor_ob_obrazovanii_480301_Teologiya_platnoe_obuchenie_inostrannye_graghdane_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/vip/Dogovor_ob_obrazovanii_000000_Podgotovka_(bakalavriat)_graghdane_RF_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/vic/Dogovor_ob_obrazovanii_000000_Podgotovka_(bakalavriat)_inostrannye_graghdane_2025.pdf",
    "https://sdamp.ru/vikon/sveden/files/aip/Prikaz_ob_ustanovlenii_stoimosti_POU_na_2025-2026_uchebnyi_god_(1_kurs).pdf",
    "https://sdamp.ru/vikon/sveden/files/rih/Prikaz_ot_30.12.2025_No_367_Ob_ustanovlenii_razmerov_GAS,_PGAS,_GSS_i_PGSS_studentam,_GS_aspirantam_i_materialynoi_podderghki_obuchayuschimsya.pdf",
    "https://sdamp.ru/vikon/sveden/files/eiw/Prikaz_ot_30.08.2024_No_166_Ob_utverghdenii_sostava_socialyno-stipendialynoi_komissii_Akademii.pdf",
    "https://sdamp.ru/vikon/sveden/files/ais/Pologhenie_o_stipendialynom_obespechenii_studentov_(ot_29-08-2024_GHurnal_(protokol)_No_8_(51).pdf",
    "https://sdamp.ru/sveden/quality/federalnyj-zakon-ot-29-dekabrya-2012-g-n-273-fz-ob-obrazovanii-v-rf.pdf",
    "https://sdamp.ru/sveden/quality/0001202009070046.pdf",
    "https://sdamp.ru/sveden/quality/МетодРеком_по_реализации_ОО_механизмов_качества.pdf",
    "http://government.ru/docs/all/109497/",
]
MAX_DEPTH = 2 # Глубина 2: Главная -> Ссылка на ней -> Ссылка на ней
MAX_PAGES = 50 # Лимит страниц, чтобы не качать весь интернет
# ================= ЛОГИКА =================
def get_links_from_page(url, domain="sdamp.ru"):
    """Ищет все ссылки на странице, ведущие внутрь академии"""
    links = set()
    try:
        # verify=False нужен, если у сайта проблемы с SSL сертификатами
        response = requests.get(url, timeout=10, verify=False)
        soup = BeautifulSoup(response.content, "html.parser")
       
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(url, href)
           
            # Фильтруем мусор и внешние ссылки
            if domain in full_url and not full_url.endswith(('.pdf', '.docx', '.jpg', '.png', '#')):
                links.add(full_url)
    except Exception as e:
        print(f"⚠️ Не удалось прочитать ссылки с {url}: {e}")
    return links
def crawl_site(start_urls, depth=1):
    """Паук для сайта"""
    visited = set()
    queue = [(url, 1) for url in start_urls]
    final_list = []
   
    print(f"🕷 Запускаю обход сайта (Глубина: {depth})...")
   
    while queue and len(visited) < MAX_PAGES:
        url, current_depth = queue.pop(0)
       
        if url in visited: continue
        visited.add(url)
        final_list.append(url)
        print(f" 🔗 Добавлена страница: {url}")
       
        if current_depth < depth:
            new_links = get_links_from_page(url)
            for link in new_links:
                if link not in visited:
                    queue.append((link, current_depth + 1))
                   
    return final_list
def create_knowledge_base():
    documents = []
   
    # 1. ОБРАБОТКА ФАЙЛОВ (с OCR)
    print(f"\n📂 1. Сканирую папку '{SOURCE_FOLDER}'...")
    if os.path.exists(SOURCE_FOLDER):
        for filename in os.listdir(SOURCE_FOLDER):
            file_path = os.path.join(SOURCE_FOLDER, filename)
            if filename.startswith("~$"): continue # Пропуск временных файлов
            try:
                if filename.lower().endswith(".pdf"):
                    print(f" 📄 PDF (OCR включен): {filename}")
                    # extract_images=True включает OCR (распознавание сканов)
                    loader = PyPDFLoader(file_path, extract_images=True)
                    documents.extend(loader.load())
                   
                elif filename.lower().endswith(".docx"):
                    print(f" 📝 Word: {filename}")
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                   
                elif filename.lower().endswith(".txt"):
                    print(f" 📜 Txt: {filename}")
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents.extend(loader.load())
            except Exception as e:
                print(f" ❌ Ошибка файла {filename}: {e}")
    else:
        print("⚠️ Папка с файлами не найдена.")
    # 2. ОБРАБОТКА САЙТА
    print(f"\n🌐 2. Сканирую сайт...")
    target_urls = crawl_site(START_URLS, MAX_DEPTH)
    print(f" ✅ Всего страниц для чтения: {len(target_urls)}")
   
    if target_urls:
        try:
            loader = WebBaseLoader(target_urls)
            loader.requests_kwargs = {'verify': False}
            web_docs = loader.load()
            documents.extend(web_docs)
            print(f" 📥 Текст с сайта загружен успешно.")
        except Exception as e:
            print(f" ❌ Ошибка загрузки сайта: {e}")
    # 3. СОЗДАНИЕ ВЕКТОРОВ
    if not documents:
        print("❌ Нечего сохранять! Проверьте файлы и интернет.")
        return
    print(f"\n🧠 3. Создаю базу знаний (всего {len(documents)} документов)...")
   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
   
    # ВАЖНО: Модель должна быть та же, что и в боте!
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
   
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
   
    print(f"\n🎉 ГОТОВО! База сохранена в папку '{DB_PATH}'")
    print("Теперь нужно отправить изменения на GitHub.")
if __name__ == "__main__":
    create_knowledge_base()
