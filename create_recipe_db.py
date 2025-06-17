import sqlite3
import json
import os
from typing import Dict, Any
import openai

def transform_recipe_with_gemini(recipe_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform recipe data using Gemini API."""
    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or OPENAI_API_KEY environment variable must be set")
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Prepare the prompt
    prompt = f"""Transform this recipe data into the specified schema:

Input Recipe:
{json.dumps(recipe_data, indent=2)}

Please transform this into a JSON object with the following fields:
- name (from title)
- ingredients (as array)
- instructions (as array of strings)
- description (synthesized from instructions)
- cuisine_type (inferred from name and ingredients)
- prep_time (estimated in minutes)
- cook_time (estimated in minutes)
- total_time (prep_time + cook_time)
- servings (estimated)
- source (always "epicurious.com")

Output the result as a JSON object only, no other text."""

    # Call Gemini API
    response = client.chat.completions.create(
        model='gemini-2.5-flash',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # Low temperature for more consistent output
    )

    # Parse the response
    try:
        # First, remove the "```json" and "```" from the response, if present
        content = response.choices[0].message.content
        if content.startswith("```json") and content.endswith("```"):
            content = content[len("```json"):-len("```")]
        transformed_data = json.loads(content)
        return transformed_data
    except json.JSONDecodeError as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        raise

def create_database(limit = 200):
    """Creates and populates the recipes.db SQLite database with transformed recipe data."""
    db_name = 'recipes.db'
    conn = None
    try:
        # Read recipes from JSON file
        with open('recipes.json', 'r', encoding='utf-8') as f:
            recipes_data = json.load(f)

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Drop the table if it already exists
        cursor.execute("DROP TABLE IF EXISTS recipes_table")

        # Create the table with the new schema
        cursor.execute('''
            CREATE TABLE recipes_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                ingredients TEXT,
                instructions TEXT,
                description TEXT,
                cuisine_type TEXT,
                prep_time INTEGER,
                cook_time INTEGER,
                total_time INTEGER,
                servings INTEGER,
                source TEXT
            )
        ''')

        # Transform and insert each recipe
        count = 0
        for recipe_id, recipe_data in recipes_data.items():
            try:
                transformed_data = transform_recipe_with_gemini(recipe_data)
                
                # Convert lists to JSON strings for storage
                ingredients_json = json.dumps(transformed_data['ingredients'])
                instructions_json = json.dumps(transformed_data['instructions'])

                cursor.execute('''
                    INSERT INTO recipes_table (
                        name, ingredients, instructions, description,
                        cuisine_type, prep_time, cook_time, total_time,
                        servings, source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transformed_data['name'],
                    ingredients_json,
                    instructions_json,
                    transformed_data['description'],
                    transformed_data['cuisine_type'],
                    transformed_data['prep_time'],
                    transformed_data['cook_time'],
                    transformed_data['total_time'],
                    transformed_data['servings'],
                    transformed_data['source']
                ))
                
                print(f"Processed recipe: {transformed_data['name']}")
                count += 1
                if count >= limit:
                    break

            except Exception as e:
                print(f"Error processing recipe {recipe_id}: {e}")
                continue

        conn.commit()
        print(f"Database '{db_name}' created and populated successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_database()
