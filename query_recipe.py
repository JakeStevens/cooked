import sqlite3
import json
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Assuming GeminiEmbeddingModel is in embedding_generator.py
from embedding_generator import GeminiEmbeddingModel

def find_similar_recipes(query: str, top_n: int = 4) -> List[Dict]:
    """
    Finds recipes similar to a given query using text embeddings.

    Args:
        query: The search query (e.g., "chicken and rice").
        top_n: The number of top similar recipes to return.

    Returns:
        A list of dictionaries, where each dictionary contains the details
        of a similar recipe.
    """
    # Initialize the embedding model
    # GeminiEmbeddingModel will look for GEMINI_API_KEY or OPENAI_API_KEY in env vars
    try:
        embed_model = GeminiEmbeddingModel()
    except ValueError as e:
        print(f"Failed to initialize GeminiEmbeddingModel: {e}")
        print("Please ensure GEMINI_API_KEY or OPENAI_API_KEY environment variable is set.")
        return []

    # Generate embedding for the query
    try:
        query_embedding = embed_model.generate_embedding(query)
    except Exception as e:
        print(f"Error generating embedding for query '{query}': {e}")
        return []
    query_embedding_np = np.array(query_embedding).reshape(1, -1)

    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('recipes.db')
        cursor = conn.cursor()

        # Fetch all recipe embeddings from the database
        cursor.execute("SELECT recipe_id, embedding_vector FROM recipe_embeddings_table")
        all_recipes_embeddings = cursor.fetchall()

        if not all_recipes_embeddings:
            print("No recipe embeddings found in the database.")
            return []

        similarities = []
        for recipe_id, embedding_json in all_recipes_embeddings:
            embedding_vector = json.loads(embedding_json)
            embedding_vector_np = np.array(embedding_vector).reshape(1, -1)

            # Calculate cosine similarity
            sim_score = cosine_similarity(query_embedding_np, embedding_vector_np)[0][0]
            similarities.append((sim_score, recipe_id))

        # Sort recipes by similarity in descending order
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Select the top_n recipes
        top_recipe_ids = [recipe_id for _, recipe_id in similarities[:top_n]]

        if not top_recipe_ids:
            return []

        # Fetch details for the top N recipes
        # Using a placeholder for the actual column names in recipes_table
        # Adjust 'name', 'ingredients', 'instructions' if your schema is different
        placeholders = ','.join('?' for _ in top_recipe_ids)
        query_details = f"SELECT recipe_id, name, ingredients, instructions, category FROM recipes_table WHERE recipe_id IN ({placeholders})"

        cursor.execute(query_details, top_recipe_ids)
        top_recipes_details = cursor.fetchall()

        # Re-order details to match similarity ranking and format as list of dicts
        ordered_recipes = []
        details_map = {row[0]: row for row in top_recipes_details} # recipe_id -> row

        for recipe_id in top_recipe_ids:
            if recipe_id in details_map:
                detail_row = details_map[recipe_id]
                # Assuming columns are: recipe_id, name, ingredients, instructions, category
                ordered_recipes.append({
                    "recipe_id": detail_row[0],
                    "name": detail_row[1],
                    "ingredients": detail_row[2],
                    "instructions": detail_row[3],
                    "category": detail_row[4],
                    "similarity_score": dict(similarities)[recipe_id] # Add similarity score
                })

        return ordered_recipes

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Example usage (requires embedding_generator.py and a populated recipes.db)

    # Test the function
    # Note: This test will only work if GEMINI_API_KEY is set in the environment
    # or if recipes.db already contains embeddings.
    print("Testing find_similar_recipes with 'pasta dish with eggs and cheese':")
    try:
        similar = find_similar_recipes("pasta dish with eggs and cheese", top_n=2)
        if similar:
            for recipe in similar:
                print(f"Found recipe: {recipe['name']} (ID: {recipe['recipe_id']}) - Score: {recipe.get('similarity_score', 'N/A')}")
                print(f"  Ingredients: {recipe['ingredients']}")
                print("-" * 20)
        elif not similar and embed_model.api_key: # Check if it was an empty result vs init failure
             print("No similar recipes found (or embeddings were not available).")
        else:
            # This path taken if embed_model init failed earlier in find_similar_recipes
            print("find_similar_recipes could not run due to embedding model initialization issues.")

    except Exception as e:
        print(f"An error occurred during find_similar_recipes test: {e}")


    print("\nTesting find_similar_recipes with 'sweet breakfast food':")
    try:
        similar_breakfast = find_similar_recipes("sweet breakfast food", top_n=2)
        if similar_breakfast:
            for recipe in similar_breakfast:
                print(f"Found recipe: {recipe['name']} (ID: {recipe['recipe_id']}) - Score: {recipe.get('similarity_score', 'N/A')}")
                print(f"  Ingredients: {recipe['ingredients']}")
                print("-" * 20)
        elif not similar_breakfast and embed_model.api_key:
            print("No similar breakfast recipes found (or embeddings were not available).")
        else:
            print("find_similar_recipes could not run due to embedding model initialization issues.")
    except Exception as e:
        print(f"An error occurred during find_similar_recipes test: {e}")
