"""
End-to-end Wikipedia RAG (Retrieval-Augmented Generation) example.

This example demonstrates:
1. Downloading Wikipedia articles
2. Chunking text into manageable pieces
3. Generating embeddings using sentence-transformers
4. Storing in PeachBase
5. Performing semantic, lexical, and hybrid search
6. Building a simple Q&A system

Requirements:
    pip install peachbase sentence-transformers wikipedia-api
"""

import re
from typing import List, Dict, Any
import wikipediaapi
from sentence_transformers import SentenceTransformer


def download_wikipedia_articles(topics: List[str], max_chars: int = 50000) -> Dict[str, str]:
    """Download Wikipedia articles for given topics.

    Args:
        topics: List of Wikipedia article titles
        max_chars: Maximum characters per article (to avoid huge downloads)

    Returns:
        Dict mapping article title to content
    """
    print(f"\n📥 Downloading Wikipedia articles...")

    # Initialize Wikipedia API
    wiki = wikipediaapi.Wikipedia(
        user_agent='PeachBase-Example/1.0',
        language='en'
    )

    articles = {}
    for topic in topics:
        print(f"  - Fetching: {topic}")
        page = wiki.page(topic)

        if page.exists():
            # Truncate if too long
            content = page.text[:max_chars]
            articles[topic] = content
            print(f"    ✓ Got {len(content)} chars")
        else:
            print(f"    ✗ Article not found")

    return articles


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    # Split into sentences first for better chunk boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))

            # Start new chunk with overlap
            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                else:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length

    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def prepare_documents(articles: Dict[str, str]) -> List[Dict[str, Any]]:
    """Chunk articles and prepare documents for PeachBase.

    Args:
        articles: Dict of article_title -> content

    Returns:
        List of document dicts with metadata
    """
    print(f"\n✂️  Chunking articles...")

    documents = []
    doc_id = 0

    for title, content in articles.items():
        # Chunk the article
        chunks = chunk_text(content, chunk_size=500, overlap=50)
        print(f"  - {title}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"doc_{doc_id}",
                "text": chunk,
                "metadata": {
                    "source": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
            doc_id += 1

    print(f"\n  Total documents: {len(documents)}")
    return documents


def generate_embeddings(documents: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    """Generate embeddings for documents using sentence-transformers.

    Args:
        documents: List of documents
        model_name: Name of sentence-transformers model

    Returns:
        Documents with embeddings added
    """
    print(f"\n🧠 Generating embeddings with {model_name}...")
    print(f"   (This may take a minute on first run - model download)")

    # Load model
    model = SentenceTransformer(model_name)
    print(f"   Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Extract texts
    texts = [doc["text"] for doc in documents]

    # Generate embeddings in batch
    print(f"   Generating embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Add embeddings to documents
    for doc, embedding in zip(documents, embeddings):
        doc["vector"] = embedding.tolist()

    print(f"   ✓ Done!")
    return documents


def create_peachbase_collection(documents: List[Dict[str, Any]], db_path: str = "./wikipedia_db"):
    """Create PeachBase collection and add documents.

    Args:
        documents: List of documents with embeddings
        db_path: Path to database

    Returns:
        Collection object
    """
    import peachbase

    print(f"\n💾 Creating PeachBase collection at {db_path}...")

    # Connect to database
    db = peachbase.connect(db_path)

    # Get embedding dimension from first document
    dimension = len(documents[0]["vector"])

    # Create collection
    collection = db.create_collection(
        name="wikipedia",
        dimension=dimension,
        overwrite=True
    )

    print(f"   Adding {len(documents)} documents...")
    collection.add(documents)

    # Save to disk
    collection.save()
    print(f"   ✓ Collection saved!")

    return collection


def search_and_display(collection, query: str, model: SentenceTransformer, mode: str = "hybrid"):
    """Perform search and display results.

    Args:
        collection: PeachBase collection
        query: Search query
        model: Embedding model
        mode: Search mode ("semantic", "lexical", or "hybrid")
    """
    print(f"\n🔍 Searching: '{query}'")
    print(f"   Mode: {mode}")
    print("   " + "=" * 70)

    # Generate query embedding for semantic/hybrid search
    if mode in ["semantic", "hybrid"]:
        query_vector = model.encode(query).tolist()
    else:
        query_vector = None

    # Perform search
    results = collection.search(
        query_text=query if mode in ["lexical", "hybrid"] else None,
        query_vector=query_vector,
        mode=mode,
        limit=5,
        alpha=0.5  # Balanced hybrid search
    )

    # Display results
    for i, result in enumerate(results.to_list(), 1):
        print(f"\n   Result {i} (Score: {result['score']:.4f})")
        print(f"   Source: {result['metadata']['source']}")
        print(f"   Chunk: {result['metadata']['chunk_index'] + 1}/{result['metadata']['total_chunks']}")
        print(f"   Text: {result['text'][:200]}...")
        if len(result['text']) > 200:
            print(f"         [...] ({len(result['text'])} chars total)")


def answer_question(collection, question: str, model: SentenceTransformer, top_k: int = 3) -> str:
    """Simple Q&A using retrieved context.

    Args:
        collection: PeachBase collection
        question: Question to answer
        model: Embedding model
        top_k: Number of context chunks to retrieve

    Returns:
        Answer string with context
    """
    print(f"\n❓ Question: {question}")
    print("   " + "=" * 70)

    # Retrieve relevant context using hybrid search
    query_vector = model.encode(question).tolist()
    results = collection.search(
        query_text=question,
        query_vector=query_vector,
        mode="hybrid",
        limit=top_k,
        alpha=0.4  # Favor semantic for Q&A
    )

    # Build context from top results
    context_parts = []
    sources = set()

    for result in results.to_list():
        context_parts.append(result['text'])
        sources.add(result['metadata']['source'])

    context = "\n\n".join(context_parts)

    print(f"\n   📚 Retrieved {len(results)} relevant passages from:")
    for source in sources:
        print(f"      - {source}")

    print(f"\n   📝 Context (for LLM to process):")
    print(f"   {'─' * 70}")
    print(f"   {context[:500]}...")
    print(f"   {'─' * 70}")

    # In a real application, you would send this to an LLM like GPT-4 or Claude
    print(f"\n   💡 In production: Send the question + context to an LLM for answer generation")

    return context


def main():
    """Main execution function."""

    print("=" * 80)
    print("🍑 PeachBase - Wikipedia RAG Example")
    print("=" * 80)

    # Step 1: Download Wikipedia articles
    topics = [
        "Machine learning",
        "Artificial intelligence",
        "Python (programming language)",
        "Vector database",
        "Natural language processing"
    ]

    articles = download_wikipedia_articles(topics, max_chars=10000)

    if not articles:
        print("\n❌ No articles downloaded. Please check your internet connection.")
        return

    # Step 2: Chunk articles
    documents = prepare_documents(articles)

    # Step 3: Generate embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional embeddings
    documents = generate_embeddings(documents, model_name="all-MiniLM-L6-v2")

    # Step 4: Create PeachBase collection
    collection = create_peachbase_collection(documents)

    # Step 5: Demonstrate different search modes
    print("\n" + "=" * 80)
    print("🎯 DEMONSTRATION: Different Search Modes")
    print("=" * 80)

    query = "What is deep learning and how does it work?"

    # Semantic search
    search_and_display(collection, query, model, mode="semantic")

    # Lexical search
    search_and_display(collection, query, model, mode="lexical")

    # Hybrid search (best of both)
    search_and_display(collection, query, model, mode="hybrid")

    # Step 6: Demonstrate Q&A
    print("\n" + "=" * 80)
    print("💬 DEMONSTRATION: Question Answering")
    print("=" * 80)

    questions = [
        "What programming language is commonly used for machine learning?",
        "How do neural networks learn?",
        "What is the difference between AI and machine learning?"
    ]

    for question in questions:
        answer_question(collection, question, model)
        print()

    # Step 7: Demonstrate metadata filtering
    print("\n" + "=" * 80)
    print("🔎 DEMONSTRATION: Metadata Filtering")
    print("=" * 80)

    print(f"\n🔍 Searching only in 'Python (programming language)' article")
    query_vector = model.encode("object oriented programming").tolist()

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        filter={"source": "Python (programming language)"},
        limit=3
    )

    print(f"\n   Found {len(results)} results:")
    for i, result in enumerate(results.to_list(), 1):
        print(f"\n   {i}. Score: {result['score']:.4f}")
        print(f"      {result['text'][:150]}...")

    # Final summary
    print("\n" + "=" * 80)
    print("✅ COMPLETE - PeachBase Wikipedia RAG Demo")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   - Downloaded {len(articles)} Wikipedia articles")
    print(f"   - Created {len(documents)} text chunks")
    print(f"   - Generated {len(documents)} embeddings")
    print(f"   - Stored in PeachBase collection: {collection.name}")
    print(f"   - Collection size: {collection.size} documents")
    print(f"\n💡 Next steps:")
    print(f"   - Connect to an LLM (GPT-4, Claude) for answer generation")
    print(f"   - Deploy to AWS Lambda for serverless RAG")
    print(f"   - Scale to thousands of Wikipedia articles")
    print(f"   - Add re-ranking for better results")
    print()


if __name__ == "__main__":
    main()
