import os
# Отключаем предупреждение о tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from sentence_transformers import SentenceTransformer

persist_dir = "chroma_db"
collection_name = "knowledge_base"

# Подключение к persistent базе
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_collection(name=collection_name)

# Проверяем количество элементов в коллекции
count = collection.count()
print("Количество элементов в коллекции:", count)

if count == 0:
    print("Коллекция пуста! Сначала загрузите embeddings.")
else:
    # Загружаем модель для query
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_text = input("Введите ваш вопрос: ")
    query_embedding = embedding_model.encode(query_text).tolist()  # вектор запроса

    # Проверяем размерность
    # Получаем первый документ из коллекции
    first_item = collection.get(limit=1)
    if first_item["embeddings"] and len(first_item["embeddings"]) > 0:
        first_embedding = first_item["embeddings"][0]
        print("Размерность embeddings в коллекции:", len(first_embedding))
    print("Размерность query_embedding:", len(query_embedding))

    # Выполняем поиск
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # Собираем найденные чанки в единый контекст
    retrieved_chunks = []
    for i in range(len(results["ids"][0])):
        retrieved_chunks.append(results["documents"][0][i])
    context_text = "\n\n---\n\n".join(retrieved_chunks)

    # Few-shot prompting с Chain-of-Thought: примеры с процессом рассуждения
    few_shot_examples = """
Пример 1:
Контекст: James Yang был студентом Health Box. Он учился в доме Brave House и был известен своими приключениями.
Вопрос: Кто такой James Yang?
Рассуждение:
1. Анализирую контекст на предмет информации о James Yang
2. В контексте упоминается: James Yang был студентом Health Box
3. Также указано: он учился в доме Brave House
4. Дополнительная информация: он был известен своими приключениями
5. Вывод: у меня есть достаточно информации для ответа
Ответ: James Yang был студентом Health Box, который учился в доме Brave House. Он известен своими приключениями.

Пример 2:
Контекст: Health Box - это школа магии, расположенная в Шотландии. Она была основана много веков назад.
Вопрос: Что такое Health Box?
Рассуждение:
1. Анализирую контекст на предмет определения Health Box
2. В контексте прямо указано: Health Box - это школа магии
3. Дополнительная информация: расположена в Шотландии, основана много веков назад
4. Вывод: у меня есть полная информация для ответа
Ответ: Health Box - это школа магии, расположенная в Шотландии. Она была основана много веков назад.

Пример 3:
Контекст: В тексте упоминается только информация о школе, но нет данных о конкретных персонах.
Вопрос: Кто был директором Health Box в 1995 году?
Рассуждение:
1. Анализирую контекст на предмет информации о директоре Health Box в 1995 году
2. В контексте упоминается только общая информация о школе
3. Конкретных данных о директоре или дате 1995 год нет
4. Вывод: информации недостаточно для ответа
Ответ: Я не знаю. В предоставленном контексте нет информации о директоре Health Box в 1995 году.

Пример 4:
Контекст: James Yang учился в Health Box. Школа была основана в средние века.
Вопрос: Какой была фамилия матери James Yang?
Рассуждение:
1. Анализирую контекст на предмет информации о матери James Yang
2. В контексте упоминается только James Yang и общая информация о школе
3. Нет никаких данных о семье, родителях или фамилии матери
4. Вывод: информации нет вообще
Ответ: Я не знаю. В предоставленном контексте нет информации о матери James Yang или её фамилии.

Пример 5:
Контекст: Health Box - это школа магии. Она расположена в Шотландии.
Вопрос: Сколько студентов училось в Health Box в 2000 году?
Рассуждение:
1. Анализирую контекст на предмет информации о количестве студентов в 2000 году
2. В контексте есть только общая информация о школе и её расположении
3. Нет никаких статистических данных или конкретных цифр о количестве студентов
4. Вывод: конкретной информации нет
Ответ: Я не знаю. В предоставленном контексте нет информации о количестве студентов в Health Box в 2000 году.
"""

    # Формируем промпт для LLM с Few-shot prompting и Chain-of-Thought
    prompt = f"""
System: Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги.

Вы ассистент, отвечающий строго на основе переданного контекста.
Используйте технику Chain-of-Thought: сначала проведите рассуждение, затем дайте ответ.

Инструкции:
1. Проанализируйте контекст на предмет релевантной информации
2. Определите, достаточно ли информации для ответа на вопрос
3. Проведите пошаговое рассуждение (всегда пишите свои шаги)
4. Дайте краткий и точный ответ на основе контекста
5. Если информации недостаточно — честно скажите об этом

{few_shot_examples}

---

Контекст:
{context_text}

Вопрос:
{query_text}

Рассуждение:
"""
    print("\n\nСформированный промпт для LLM:\n")
    print(prompt)

    # Отправляем промпт в локальную модель GPT4All
    from gpt4all import GPT4All

    # Попробуем использовать модель по умолчанию или указать правильное имя
    # Популярные модели GPT4All: "mistral-7b-openorca.Q4_0.gguf", "orca-mini-3b.gguf", "llama-2-7b-chat.gguf"
    # Если модель не найдена, GPT4All попытается скачать её автоматически
    
    try:
        # Сначала пробуем использовать модель по умолчанию (если уже установлена)
        print("\nИнициализация GPT4All модели...")
        llm_model = GPT4All("mistral-7b-openorca.Q4_0.gguf", allow_download=True, verbose=True)
        # Увеличиваем max_tokens для Chain-of-Thought (рассуждение + ответ)
        answer = llm_model.generate(prompt, max_tokens=800)
    except Exception as e:
        print(f"\nОшибка при загрузке модели: {e}")
        print("Попробуем использовать другую модель...")
        try:
            # Пробуем более легкую модель
            llm_model = GPT4All("orca-mini-3b.gguf", allow_download=True, verbose=True)
            # Увеличиваем max_tokens для Chain-of-Thought (рассуждение + ответ)
            answer = llm_model.generate(prompt, max_tokens=800)
        except Exception as e2:
            print(f"\nОшибка при загрузке альтернативной модели: {e2}")
            print("\nИспользуем только результаты поиска без LLM:")
            answer = "Не удалось загрузить LLM модель. См. результаты поиска выше."
    print("\nОтвет модели (GPT4All локально):\n")
    print(answer)

    # Вывод результатов
    for i in range(len(results["ids"][0])):
        print(f"\nID: {results['ids'][0][i]}")
        print(f"Source file: {results['metadatas'][0][i]['source_file']}")
        print(f"Chunk text: {results['documents'][0][i]}")