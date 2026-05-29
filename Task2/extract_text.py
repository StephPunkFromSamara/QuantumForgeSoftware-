import os
import re
import json
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Настройка словаря замен
# Если нет файла terms_map.json — создаётся пустой
# -----------------------------
TERMS_FILE = "terms_map.json"

if os.path.exists(TERMS_FILE):
    with open(TERMS_FILE, encoding="utf-8") as f:
        terms_map = json.load(f)
else:
    terms_map = {
        "Darth Vader": "Xarn Velgor",
        "Death Star": "Void Core",
        "The Force": "Synth Flux"
        # добавь свои замены здесь
    }
    with open(TERMS_FILE, "w", encoding="utf-8") as f:
        json.dump(terms_map, f, indent=4, ensure_ascii=False)

# -----------------------------
# Функции
# -----------------------------
def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def extract_sections(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    sections = []
    current_title = "intro"
    current_text = []
    for elem in soup.find_all(["h1","h2","h3","p","li"]):
        if elem.name in ["h1","h2","h3"]:
            if current_text:
                sections.append((current_title, "\n".join(current_text)))
                current_text = []
            current_title = elem.get_text().strip()
        else:
            current_text.append(elem.get_text())
    if current_text:
        sections.append((current_title, "\n".join(current_text)))
    return sections

def safe_name(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def replace_terms(text, terms_map):
    for k, v in terms_map.items():
        # заменяем все вхождения слова с учетом границ
        text = re.sub(rf"\b{k}\b", v, text)
    return text

# -----------------------------
# Основной блок
# -----------------------------


urls = [
    "https://harrypotter.fandom.com/wiki/Harry_Potter",
    "https://harrypotter.fandom.com/wiki/Tom_Riddle",
    "https://harrypotter.fandom.com/wiki/Severus_Snape",
    "https://harrypotter.fandom.com/wiki/James_Potter_(I)",
    "https://harrypotter.fandom.com/wiki/First_Wizarding_War",
    "https://harrypotter.fandom.com/wiki/Albus_Dumbledore",
    "https://harrypotter.fandom.com/wiki/Hogwarts_School_of_Witchcraft_and_Wizardry",
    "https://harrypotter.fandom.com/wiki/Rubeus_Hagrid",
    "https://harrypotter.fandom.com/wiki/Philosopher%27s_Stone",
    "https://harrypotter.fandom.com/wiki/Cedric_Diggory",
    "https://harrypotter.fandom.com/wiki/Chamber_of_Secrets",
    "https://harrypotter.fandom.com/wiki/Ronald_Weasley",
    "https://harrypotter.fandom.com/wiki/Hermione_Granger",
    "https://harrypotter.fandom.com/wiki/Minerva_McGonagall",
    "https://harrypotter.fandom.com/wiki/Battle_of_the_Department_of_Mysteries",
    "https://harrypotter.fandom.com/wiki/Sirius_Black",
    "https://harrypotter.fandom.com/wiki/King%27s_Cross_Station",
    "https://harrypotter.fandom.com/wiki/Platform_Nine_and_Three-Quarters",
    "https://harrypotter.fandom.com/wiki/Hogwarts_Express",
    "https://harrypotter.fandom.com/wiki/Draco_Malfoy",
    "https://harrypotter.fandom.com/wiki/Gryffindor_Tower",
    "https://harrypotter.fandom.com/wiki/Argus_Filch",
    "https://harrypotter.fandom.com/wiki/Gregory_Goyle",
    "https://harrypotter.fandom.com/wiki/Gryffindor",
    "https://harrypotter.fandom.com/wiki/Triwizard_Tournament",
    "https://harrypotter.fandom.com/wiki/Deathly_Hallows",
    "https://harrypotter.fandom.com/wiki/Helga_Hufflepuff",
    "https://harrypotter.fandom.com/wiki/Rowena_Ravenclaw",
    "https://harrypotter.fandom.com/wiki/Godric_Gryffindor",
    "https://harrypotter.fandom.com/wiki/Phoenix"
]

os.makedirs("knowledge_base_raw", exist_ok=True)

for url in urls:
    print("Processing:", url)
    response = requests.get(url)
    response.raise_for_status()
    html = response.text

    sections = extract_sections(html)
    page_name = safe_name(url.split("/")[-1])

    for i, (title, text) in enumerate(sections, 1):
        # заменяем ключевые термины
        text = replace_terms(text, terms_map)

        filename = f"{i:02d}_{page_name}_{safe_name(title)}.txt"
        filepath = os.path.join("knowledge_base_raw", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

print("Done! Все тексты сохранены в папке knowledge_base_raw/")