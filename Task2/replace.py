import os
import json
import re
from slugify import slugify

# -----------------------------
# Папки
# -----------------------------
RAW_FOLDER = "knowledge_base_raw"
OUTPUT_FOLDER = "knowledge_base"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Загружаем словарь замен
# -----------------------------
TERMS_FILE = "terms_map.json"
with open(TERMS_FILE, encoding="utf-8") as f:
    terms_map = json.load(f)

# -----------------------------
# Функция замены терминов в тексте
# -----------------------------
def replace_terms_partial(text, terms_map):
    for k, v in terms_map.items():
        text = text.replace(k, v)
    return text

# -----------------------------
# Функция создания безопасного имени файла
# -----------------------------
def anonymize_filename(filename, terms_map):
    # убираем расширение
    name, ext = os.path.splitext(filename)
    # заменяем термины на новые
    for k, v in terms_map.items():
        name = name.replace(k, v)
    # делаем безопасный slug
    name = slugify(name)
    return f"{name}{ext}"

# -----------------------------
# Обработка всех файлов
# -----------------------------
for file in os.listdir(RAW_FOLDER):
    if file.endswith(".txt"):
        input_path = os.path.join(RAW_FOLDER, file)

        # Читаем текст
        with open(input_path, encoding="utf-8") as f:
            text = f.read()

        # Замена терминов в тексте
        text = replace_terms_partial(text, terms_map)

        # Создаём новое имя файла
        new_file = anonymize_filename(file, terms_map)
        output_path = os.path.join(OUTPUT_FOLDER, new_file)

        # Сохраняем текст
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Processed {file} -> {new_file}")

print("Done! Все тексты и имена файлов анонимизированы и сохранены в", OUTPUT_FOLDER)