import os
import chromadb

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from database import get_products

load_dotenv()

model = SentenceTransformer(
    os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
)

client = chromadb.PersistentClient(
    path=os.getenv("CHROMA_PATH", "./chroma_db")
)

collection = client.get_or_create_collection(
    name="unimart_products",
    metadata={"hnsw:space": "cosine"}
)


def index_products():
    products = get_products()

    if not products:
        print("No products found.")
        return

    documents = []
    ids = []
    metadata = []

    for product in products:
        product_id = str(product["id"])

        content = f"""
        Product Name: {product['name']}
        Description: {product['description'] or ''}
        Category: {product['category']}
        Price: Rs. {product['price']}
        Stock: {product['stock']}
        """

        documents.append(content)
        ids.append(product_id)

        metadata.append({
            "product_id": product_id,
            "name": str(product["name"]),
            "category": str(product["category"] or "")
        })

    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata
    )

    # Remove products that no longer exist in MySQL.
    current_ids = set(ids)
    stored = collection.get(include=[])
    stale_ids = [
        product_id
        for product_id in stored["ids"]
        if product_id not in current_ids
    ]

    if stale_ids:
        collection.delete(ids=stale_ids)

    print(f"Indexed {len(products)} products successfully.")


if __name__ == "__main__":
    index_products()
