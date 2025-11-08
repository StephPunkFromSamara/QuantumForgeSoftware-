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
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_text = "Where did Harry Potter go after the battle?"
    query_embedding = model.encode(query_text).tolist()  # вектор запроса

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

    # Вывод результатов
    for i in range(len(results["ids"][0])):
        print(f"\nID: {results['ids'][0][i]}")
        print(f"Source file: {results['metadatas'][0][i]['source_file']}")
        print(f"Chunk text: {results['documents'][0][i]}")