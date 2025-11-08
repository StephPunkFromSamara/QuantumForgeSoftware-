import json
import os
import chromadb

persist_dir = "chroma_db"
collection_name = "knowledge_base"
os.makedirs(persist_dir, exist_ok=True)
json_file = "embeddings_dataset.json"

# Подключение к persistent базе данных
client = chromadb.PersistentClient(path=persist_dir)

# Удаляем существующую коллекцию, если она есть (для перезаливки)
if collection_name in [c.name for c in client.list_collections()]:
    print(f"Удаление существующей коллекции '{collection_name}'...")
    client.delete_collection(name=collection_name)
    print("Коллекция удалена.")

# Создаем новую коллекцию
print(f"Создание новой коллекции '{collection_name}'...")
collection = client.create_collection(name=collection_name)

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# -----------------------------
# Создание списков
# -----------------------------
ids = [f"{item['title']}_chunk{item['chunk_id']}" for item in data]
texts = [item["chunk_text"] for item in data]
embeddings = [item["embedding"] for item in data]
metadatas = [{
    "source_file": item.get("source_file", "unknown"),
    "title": item.get("title", "unknown"),
    "chunk_id": item.get("chunk_id"),
    "start_pos": item.get("start_pos"),
    "end_pos": item.get("end_pos"),
    "word_count": item.get("word_count")
} for item in data]

# -----------------------------
# Добавление по батчам
# -----------------------------
batch_size = 1000
for i in range(0, len(ids), batch_size):
    collection.add(
        ids=ids[i:i+batch_size],
        documents=texts[i:i+batch_size],
        embeddings=embeddings[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size]
    )
    print(f"Добавлено батчей: {i} - {i+batch_size}")

print(f"\nЗагружено {len(ids)} чанков в коллекцию '{collection_name}'!")

# -----------------------------
# Валидация
# -----------------------------
stored_ids = collection.get()["ids"]
stored_count = len(stored_ids)
print(f"\nВ коллекции реально хранится: {stored_count} документов")

# Покажем первые 5 для проверки
if stored_count > 0:
    results = collection.get(ids=stored_ids[:5])
    for i in range(len(results["ids"])):
        print("\nID:", results["ids"][i])
        print("Text:", results["documents"][i])
        print("Metadata:", results["metadatas"][i])