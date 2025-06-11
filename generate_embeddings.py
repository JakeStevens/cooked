import sqlite3
import random
import json

def generate_random_embedding_vector(dimensions=32):
    """Generates a list of 'dimensions' random floating-point numbers between 0.0 and 1.0."""
    return [random.random() for _ in range(dimensions)]

def create_embeddings():
    """
    Connects to recipes.db, generates embeddings for recipes that don't have them,
    and stores them in recipe_embeddings_table.
    """
    db_name = 'recipes.db'
    conn = None
    new_embeddings_count = 0

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create the recipe_embeddings_table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipe_embeddings_table (
                recipe_id INTEGER PRIMARY KEY,
                embedding_vector TEXT,
                FOREIGN KEY (recipe_id) REFERENCES recipes_table(id)
            )
        ''')
        conn.commit() # Commit table creation

        # Fetch all recipe IDs from recipes_table
        cursor.execute("SELECT id FROM recipes_table")
        all_recipe_ids = cursor.fetchall() # Returns a list of tuples, e.g., [(1,), (2,)]

        for row in all_recipe_ids:
            recipe_id = row[0]

            # Check if the recipe_id already exists in recipe_embeddings_table
            cursor.execute("SELECT recipe_id FROM recipe_embeddings_table WHERE recipe_id = ?", (recipe_id,))
            existing_embedding = cursor.fetchone()

            if existing_embedding is None:
                # Generate a new embedding
                embedding_vector = generate_random_embedding_vector()
                # Convert the embedding list to a JSON string
                embedding_json = json.dumps(embedding_vector)

                # Insert the recipe_id and its JSON string embedding
                cursor.execute('''
                    INSERT INTO recipe_embeddings_table (recipe_id, embedding_vector)
                    VALUES (?, ?)
                ''', (recipe_id, embedding_json))
                new_embeddings_count += 1

        conn.commit()
        print(f"Generated and added {new_embeddings_count} new embeddings to 'recipe_embeddings_table'.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        if conn: # Rollback changes if an error occurs during transaction
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_embeddings()
