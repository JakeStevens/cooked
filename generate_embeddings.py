import sqlite3
import json
from types import Recipe
from embedding_generator import generate_embedding_vector, QwenEmbeddingModel

def create_recipe_from_db(cursor, recipe_id: int) -> Recipe:
    """Create a Recipe object from database data."""
    cursor.execute("""
        SELECT id, name, ingredients, instructions, description, cuisine_type,
               prep_time, cook_time, total_time, servings
        FROM recipes_table WHERE id = ?
    """, (recipe_id,))
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Recipe with id {recipe_id} not found")
    
    # Assuming ingredients and instructions are stored as JSON strings in the database
    ingredients = json.loads(row[2])
    instructions = json.loads(row[3])
    
    return Recipe(
        id=row[0],
        name=row[1],
        ingredients=ingredients,
        instructions=instructions,
        description=row[4],
        cuisine_type=row[5],
        prep_time=row[6],
        cook_time=row[7],
        total_time=row[8],
        servings=row[9]
    )

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
        conn.commit()

        # Initialize the embedding model
        model = QwenEmbeddingModel()

        # Fetch all recipe IDs from recipes_table
        cursor.execute("SELECT id FROM recipes_table")
        all_recipe_ids = cursor.fetchall()

        for row in all_recipe_ids:
            recipe_id = row[0]

            # Check if the recipe_id already exists in recipe_embeddings_table
            cursor.execute("SELECT recipe_id FROM recipe_embeddings_table WHERE recipe_id = ?", (recipe_id,))
            existing_embedding = cursor.fetchone()

            if existing_embedding is None:
                # Create Recipe object from database data
                recipe = create_recipe_from_db(cursor, recipe_id)
                
                # Generate embedding using the new function
                embedding_vector = generate_embedding_vector(recipe, model)
                
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
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_embeddings()
