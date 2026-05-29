import os
import warnings
# Отключаем предупреждение о tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Подавляем предупреждения об устаревших моделях GPT4All
warnings.filterwarnings("ignore", category=UserWarning, message=".*out-of-date.*")

import chromadb
from sentence_transformers import SentenceTransformer

# Режим debug (по умолчанию False для обычного пользователя)
# Для включения debug режима в REPL: DEBUG = True
DEBUG = False

persist_dir = "chroma_db"
collection_name = "knowledge_base"

# Подключение к persistent базе
if DEBUG:
    print(f"[DEBUG] Подключение к базе данных: {persist_dir}")
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_collection(name=collection_name)

# Проверяем количество элементов в коллекции
count = collection.count()
if DEBUG:
    print(f"Количество элементов в коллекции: {count}")
    print(f"[DEBUG] Коллекция '{collection_name}' успешно загружена")

if count == 0:
    print("Коллекция пуста! Сначала загрузите embeddings.")
else:
    # Загружаем модель для query (один раз при запуске)
    if DEBUG:
        print("[DEBUG] Загрузка модели эмбеддингов: all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    if DEBUG:
        print("[DEBUG] Модель эмбеддингов загружена успешно")
    
    # Инициализируем LLM модель один раз при запуске
    from gpt4all import GPT4All
    
    # Популярные модели GPT4All: "mistral-7b-openorca.Q4_0.gguf", "orca-mini-3b.gguf", "llama-2-7b-chat.gguf"
    # GPT4All автоматически кеширует модели в ~/.cache/gpt4all/
    # Проверяем наличие модели локально перед скачиванием
    
    # Список моделей для попытки загрузки (в порядке приоритета)
    model_names = ["mistral-7b-openorca.Q4_0.gguf", "orca-mini-3b.gguf"]
    
    # Проверяем наличие моделей локально
    cache_dir = os.path.expanduser("~/.cache/gpt4all/")
    os.makedirs(cache_dir, exist_ok=True)
    
    llm_model = None
    model_used = None
    
    if DEBUG:
        print("\n[DEBUG] Инициализация GPT4All модели...")
        print(f"[DEBUG] Кеш моделей: {cache_dir}")
    
    # Проверяем наличие моделей в кеше
    for model_name in model_names:
        model_path = os.path.join(cache_dir, model_name)
        
        if os.path.exists(model_path):
            if DEBUG:
                print(f"[DEBUG] ✓ Найдена локальная модель: {model_name}")
                print(f"[DEBUG]   Путь: {model_path}")
            try:
                # Используем локальную модель (не скачиваем)
                llm_model = GPT4All(model_path, verbose=DEBUG)
                model_used = model_name
                if DEBUG:
                    print(f"[DEBUG] ✓ Модель {model_name} загружена из кеша")
                break
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] ⚠ Ошибка при загрузке из кеша: {e}")
                continue
    
    # Если модель не найдена локально, пробуем скачать
    if llm_model is None:
        if DEBUG:
            print("[DEBUG] Локальная модель не найдена. Попытка скачать...")
        
        for model_name in model_names:
            try:
                if DEBUG:
                    print(f"[DEBUG] Скачивание модели: {model_name}...")
                # GPT4All автоматически скачает и закеширует модель
                llm_model = GPT4All(model_name, allow_download=True, verbose=DEBUG)
                model_used = model_name
                if DEBUG:
                    print(f"[DEBUG] ✓ Модель {model_name} успешно скачана и сохранена в {cache_dir}")
                break
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] ⚠ Ошибка при скачивании {model_name}: {e}")
                continue
    
    if llm_model is None:
        print("⚠ Не удалось загрузить LLM модель. Бот будет работать только с поиском.")
    
    # Основной цикл бота
    print("\n" + "="*60)
    print("Бот готов к работе! Введите 'exit' или 'quit' для выхода.")
    print("="*60 + "\n")
    
    while True:
        query_text = input("Введите ваш вопрос: ").strip()
        
        # Проверка на выход
        if not query_text or query_text.lower() in ['exit', 'quit', 'выход']:
            print("\nДо свидания!")
            break
        
        if DEBUG:
            print(f"[DEBUG] Получен вопрос пользователя: {query_text}")
            print("[DEBUG] Создание эмбеддинга для запроса...")
        
        query_embedding = embedding_model.encode(query_text).tolist()  # вектор запроса
        if DEBUG:
            print(f"[DEBUG] Эмбеддинг создан, размерность: {len(query_embedding)}")

        # Проверяем размерность (только в debug режиме)
        if DEBUG:
            print("[DEBUG] Проверка размерности эмбеддингов в коллекции...")
            first_item = collection.get(limit=1)
            if first_item["embeddings"] and len(first_item["embeddings"]) > 0:
                first_embedding = first_item["embeddings"][0]
                print(f"[DEBUG] Размерность эмбеддингов в коллекции: {len(first_embedding)}")
                print(f"[DEBUG] Размерность query эмбеддинга: {len(query_embedding)}")
                if len(first_embedding) == len(query_embedding):
                    print("[DEBUG] ✓ Размерности совпадают")
                else:
                    print(f"[DEBUG] ⚠ Размерности не совпадают!")

        # Выполняем поиск
        if DEBUG:
            print("[DEBUG] Выполнение поиска в векторной базе данных...")
            print(f"[DEBUG] Параметры поиска: n_results=5")
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        if DEBUG:
            print(f"[DEBUG] Найдено результатов: {len(results['ids'][0])}")
            print(f"[DEBUG] IDs найденных документов: {results['ids'][0]}")

        # Собираем найденные чанки в единый контекст
        if DEBUG:
            print("[DEBUG] Формирование контекста из найденных чанков...")
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append(results["documents"][0][i])
            if DEBUG:
                chunk_length = len(results["documents"][0][i])
                print(f"[DEBUG] Чанк {i+1}: ID={results['ids'][0][i]}, длина={chunk_length} символов")
        
        context_text = "\n\n---\n\n".join(retrieved_chunks)
        if DEBUG:
            print(f"[DEBUG] Общая длина контекста: {len(context_text)} символов")
            print(f"[DEBUG] Количество чанков в контексте: {len(retrieved_chunks)}")

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
        if DEBUG:
            print("\n[DEBUG] Сформированный промпт для LLM:")
            print("="*60)
            print(prompt)
            print("="*60)
            print(f"[DEBUG] Длина промпта: {len(prompt)} символов")

        # Отправляем промпт в локальную модель GPT4All
        if llm_model is None:
            if DEBUG:
                print("[DEBUG] ⚠ LLM модель не загружена, используем только результаты поиска")
            answer = "Не удалось загрузить LLM модель. См. результаты поиска выше."
        else:
            if DEBUG:
                print(f"[DEBUG] Используется модель: {model_used}")
            
            # Генерируем ответ
            if DEBUG:
                print("[DEBUG] Генерация ответа с max_tokens=800...")
            answer = llm_model.generate(prompt, max_tokens=800)
            if DEBUG:
                print(f"[DEBUG] ✓ Ответ сгенерирован, длина: {len(answer)} символов")
        
        # Выводим только рассуждение и ответ (без лишней информации)
        print("\n" + answer)
        
        if DEBUG:
            print(f"\n[DEBUG] Длина ответа: {len(answer)} символов")
            print("\n[DEBUG] Вывод детальной информации о найденных документах:")
            print("="*60)
            for i in range(len(results["ids"][0])):
                print(f"\nID: {results['ids'][0][i]}")
                print(f"Source file: {results['metadatas'][0][i]['source_file']}")
                print(f"Metadata: {results['metadatas'][0][i]}")
                print(f"Distance: {results['distances'][0][i] if 'distances' in results else 'N/A'}")
                print(f"Chunk text: {results['documents'][0][i]}")
            print("="*60)
        
        print()  # Пустая строка для разделения между вопросами