"""
AWS Lambda deployment example for PeachBase.

Shows how to use PeachBase in a Lambda function with S3 storage
for fast, serverless vector search.
"""

import json
import peachbase


# Lambda handler function
def lambda_handler(event, context):
    """
    AWS Lambda handler for PeachBase search API.

    Expected event format:
    {
        "action": "search",  # or "add_documents"
        "collection": "my_collection",
        "query_text": "optional text query",
        "query_vector": [0.1, 0.2, ...],  # optional vector
        "mode": "semantic",  # or "lexical", "hybrid"
        "limit": 10,
        "filter": {"category": "tech"}  # optional
    }
    """

    try:
        # Parse request
        action = event.get("action")
        collection_name = event.get("collection")

        if not collection_name:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "collection name required"})
            }

        # Connect to S3-backed database
        # Collections are stored in S3 for persistence
        db = peachbase.connect("s3://my-bucket/peachbase")

        if action == "search":
            # Open existing collection
            collection = db.open_collection(collection_name)

            # Extract search parameters
            query_text = event.get("query_text")
            query_vector = event.get("query_vector")
            mode = event.get("mode", "semantic")
            limit = event.get("limit", 10)
            filter_query = event.get("filter")

            # Perform search
            results = collection.search(
                query_text=query_text,
                query_vector=query_vector,
                mode=mode,
                limit=limit,
                filter=filter_query
            )

            # Return results
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "results": [
                        {
                            "id": r["id"],
                            "text": r["text"],
                            "score": r["score"],
                            "metadata": r["metadata"]
                        }
                        for r in results.to_list()
                    ],
                    "count": len(results)
                })
            }

        elif action == "add_documents":
            # Create or open collection
            dimension = event.get("dimension", 384)
            try:
                collection = db.open_collection(collection_name)
            except Exception:
                collection = db.create_collection(collection_name, dimension=dimension)

            # Add documents
            documents = event.get("documents", [])
            if documents:
                collection.add(documents)
                collection.save()  # Persist to S3

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": f"Added {len(documents)} documents",
                    "collection_size": collection.size
                })
            }

        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Unknown action: {action}"})
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


# For local testing
def test_local():
    """Test Lambda function locally."""

    # Test search request
    search_event = {
        "action": "search",
        "collection": "articles",
        "query_vector": [0.1] * 384,  # Mock vector
        "mode": "semantic",
        "limit": 5
    }

    response = lambda_handler(search_event, None)
    print("Search Response:")
    print(json.dumps(json.loads(response["body"]), indent=2))


if __name__ == "__main__":
    test_local()
