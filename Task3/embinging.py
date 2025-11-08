import os
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# -----------------------------
# Папки и файлы
# -----------------------------
INPUT_FOLDER = "knowledge_base"  # анонимизированные тексты
OUTPUT_FILE = "embeddings_dataset.json"

os.makedirs(INPUT_FOLDER, exist_ok=True)

# -----------------------------
# Настройка текстового сплиттера
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # слова
    chunk_overlap=50       # перекрытие слов
)

# -----------------------------
# Инициализация локальной модели
# -----------------------------
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# -----------------------------
# Функции
# -----------------------------
def count_words(text):
    return len(re.findall(r'\w+', text))

def get_title(file_name):
    return os.path.splitext(file_name)[0]

# -----------------------------
# Основной пайплайн
# -----------------------------
dataset = []

for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".txt"):
        file_path = os.path.join(INPUT_FOLDER, file)
        title = get_title(file)

        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        # Разбиваем текст на чанки
        chunks = text_splitter.split_text(text)

        # Получаем позиции каждого чанка в исходном тексте
        start_pos = 0
        for i, chunk_text in enumerate(chunks):
            end_pos = start_pos + len(chunk_text)

            embedding_vector = model.encode(chunk_text).tolist()
            word_count = count_words(chunk_text)

            dataset.append({
                "source_file": file_path,
                "title": title,
                "chunk_id": i,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "word_count": word_count,
                "chunk_text": chunk_text,
                "embedding": embedding_vector
            })

            start_pos += len(chunk_text) - 50  # перекрытие слов учтено грубо

        print(f"Processed {file} into {len(chunks)} chunks.")

# -----------------------------
# Сохраняем результат
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("Done! Embeddings dataset saved to", OUTPUT_FILE)