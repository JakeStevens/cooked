import sqlite3
import json
from types import Recipe
from embedding_generator import generate_embedding_vector, GeminiEmbeddingModel
import openai
import os

def generate_recipe_overview(recipe: Recipe, client) -> str:
    """Generate a comprehensive overview for a recipe using LLM."""
    prompt = f"""Your task is to generate a comprehensive "Overview" section for a recipe, based on the provided recipe schema. This overview should be a concise, engaging paragraph (or a few closely related paragraphs) that highlights key aspects of the dish, making it easy for a user to understand its suitability and appeal.

Recipe Information:
Name: {recipe.name}
Description: {recipe.description}
Cuisine Type: {recipe.cuisine_type}
Ingredients: {json.dumps(recipe.ingredients, indent=2)}
Instructions: {json.dumps(recipe.instructions, indent=2)}
Prep Time: {recipe.prep_time} minutes
Cook Time: {recipe.cook_time} minutes
Total Time: {recipe.total_time} minutes
Servings: {recipe.servings}

Please generate an overview that includes:
- Pairings/Serving Suggestions
- Dietary Information/Restrictions
- Ease of Cooking
- Time Commitment
- Versatility/Customization
- Flavor Profile/Texture

Output the result as a JSON object with a single "overview" field containing the text."""

    response = client.chat.completions.create(
        model='gemini-2.5-flash',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    try:
        content = response.choices[0].message.content
        if content.startswith("```json") and content.endswith("```"):
            content = content[len("```json"):-len("```")]
        result = json.loads(content)
        return result["overview"]
    except json.JSONDecodeError as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        raise

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

def format_recipe_text_with_overview(recipe: Recipe, overview: str) -> str:
    """Convert a Recipe object and its overview into a single string for embedding generation."""
    components = [
        f"Recipe: {recipe.name}",
        f"Description: {recipe.description or ''}",
        f"Cuisine Type: {recipe.cuisine_type or ''}",
        "Ingredients:",
        *[f"- {ingredient}" for ingredient in recipe.ingredients],
        "Instructions:",
        *[f"{i+1}. {instruction}" for i, instruction in enumerate(recipe.instructions)],
        f"Prep Time: {recipe.prep_time or 'N/A'} minutes",
        f"Cook Time: {recipe.cook_time or 'N/A'} minutes",
        f"Total Time: {recipe.total_time or 'N/A'} minutes",
        f"Servings: {recipe.servings or 'N/A'}",
        "\nOverview:",
        overview
    ]
    return "\n".join(components)

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

        # Initialize the embedding model and Gemini client
        model = GeminiEmbeddingModel()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or OPENAI_API_KEY environment variable must be set")
        
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

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
                
                # Generate overview using Gemini
                overview = generate_recipe_overview(recipe, client)
                
                # Generate embedding using the recipe text with overview
                recipe_text = format_recipe_text_with_overview(recipe, overview)
                embedding_vector = model.generate_embedding(recipe_text)
                
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
