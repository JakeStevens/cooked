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
        print(f"Found {len(all_recipes_embeddings)} recipe embeddings in the database.")
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
            print("[DEBUG] No top recipe IDs found after similarity ranking. Returning empty result.")
            return []

        # Ensure all IDs are plain Python int
        top_recipe_ids = [int(rid) for rid in top_recipe_ids]
        print(f"[DEBUG] Querying details for recipe IDs: {top_recipe_ids}")

        # Fetch details for the top N recipes
        # Use correct column names: id, name, ingredients, instructions, description, cuisine_type, prep_time, cook_time, total_time, servings, source
        placeholders = ','.join('?' for _ in top_recipe_ids)
        query_details = f"SELECT id, name, ingredients, instructions, description, cuisine_type, prep_time, cook_time, total_time, servings, source FROM recipes_table WHERE id IN ({placeholders})"
        print(f"[DEBUG] Querying details for {len(top_recipe_ids)} recipes: {query_details}")
        try:
            cursor.execute(query_details, top_recipe_ids)
            top_recipes_details = cursor.fetchall()
        except Exception as e:
            print(f"[ERROR] Exception during recipe details fetch: {e}")
            print(f"[ERROR] Query: {query_details}")
            print(f"[ERROR] Params: {top_recipe_ids}")
            return []

        # Re-order details to match similarity ranking and format as list of dicts
        ordered_recipes = []
        details_map = {row[0]: row for row in top_recipes_details} # id -> row
        sim_map = {recipe_id: sim_score for sim_score, recipe_id in similarities} # recipe_id -> sim_score

        for recipe_id in top_recipe_ids:
            if recipe_id in details_map:
                detail_row = details_map[recipe_id]
                # Columns: id, name, ingredients, instructions, description, cuisine_type, prep_time, cook_time, total_time, servings, source
                ordered_recipes.append({
                    "recipe_id": detail_row[0],
                    "name": detail_row[1],
                    "ingredients": detail_row[2],
                    "instructions": detail_row[3],
                    "description": detail_row[4],
                    "cuisine_type": detail_row[5],
                    "prep_time": detail_row[6],
                    "cook_time": detail_row[7],
                    "total_time": detail_row[8],
                    "servings": detail_row[9],
                    "source": detail_row[10],
                    "similarity_score": sim_map[recipe_id]
                })

        return ordered_recipes

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error of type {type(e)} occurred: {e}")
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
        else:
            print("No similar recipes found (or embeddings were not available, or embedding model failed to initialize).")
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
        else:
            print("No similar breakfast recipes found (or embeddings were not available, or embedding model failed to initialize).")
    except Exception as e:
        print(f"An error occurred during find_similar_recipes test: {e}")
