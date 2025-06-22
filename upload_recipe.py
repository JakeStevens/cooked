import sqlite3
import json
import os
import base64
from typing import Dict, Any, Optional
import openai
from PIL import Image
import io
from cooked_types import Recipe

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_recipe_from_image(image_path: str) -> Recipe:
    """Parse recipe from image using Gemini Vision API and return a Recipe object."""
    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or OPENAI_API_KEY environment variable must be set")
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the prompt for recipe parsing
    prompt = """Please analyze this recipe image and extract the recipe information. 
    
    Return a JSON object with the following structure:
    {
        "name": "Recipe name",
        "ingredients": ["ingredient 1", "ingredient 2", ...],
        "instructions": ["step 1", "step 2", ...],
        "description": "Brief description of the recipe",
        "cuisine_type": "Type of cuisine (e.g., Italian, Mexican, etc.)",
        "prep_time": 15,
        "cook_time": 30,
        "total_time": 45,
        "servings": 4,
        "source": "Cookbook name and author, where possible"
    }
    
    Please estimate prep_time, cook_time, and servings if not explicitly stated.
    Return only the JSON object, no additional text."""

    # Call Gemini Vision API
    response = client.chat.completions.create(
        model='gemini-2.5-flash',
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.1  # Low temperature for more consistent output
    )

    # Parse the response
    try:
        content = response.choices[0].message.content
        # Clean up the response if it contains markdown formatting
        if content.startswith("```json") and content.endswith("```"):
            content = content[len("```json"):-len("```")]
        elif content.startswith("```") and content.endswith("```"):
            content = content[len("```"):-len("```")]
        
        parsed_data = json.loads(content)
        
        # Create Recipe object
        recipe = Recipe(
            id=0,  # Will be set by database
            name=parsed_data['name'],
            ingredients=parsed_data['ingredients'],
            instructions=parsed_data['instructions'],
            description=parsed_data.get('description'),
            cuisine_type=parsed_data.get('cuisine_type'),
            prep_time=parsed_data.get('prep_time'),
            cook_time=parsed_data.get('cook_time'),
            total_time=parsed_data.get('total_time'),
            servings=parsed_data.get('servings'),
            source=parsed_data.get('source', 'image_upload')
        )
        
        return recipe
    except json.JSONDecodeError as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        raise

def store_recipe_in_database(recipe: Recipe) -> int:
    """Store a Recipe object in the SQLite database and return the recipe ID."""
    db_name = 'recipes.db'
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Convert lists to JSON strings for storage
        ingredients_json = json.dumps(recipe.ingredients)
        instructions_json = json.dumps(recipe.instructions)

        cursor.execute('''
            INSERT INTO recipes_table (
                name, ingredients, instructions, description,
                cuisine_type, prep_time, cook_time, total_time,
                servings, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            recipe.name,
            ingredients_json,
            instructions_json,
            recipe.description,
            recipe.cuisine_type,
            recipe.prep_time,
            recipe.cook_time,
            recipe.total_time,
            recipe.servings,
            recipe.source
        ))
        
        recipe_id = cursor.lastrowid
        conn.commit()
        print(f"Recipe '{recipe.name}' stored with ID: {recipe_id}")
        return recipe_id

    except Exception as e:
        print(f"Error storing recipe in database: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def upload_recipe_from_image(image_path: str, dry_run: bool = False) -> Optional[Recipe]:
    """
    Main function to upload a recipe from an image.
    
    Args:
        image_path: Path to the PNG, SVG, or JPG image file
        dry_run: If True, only parse and print the recipe without storing to database
        
    Returns:
        The Recipe object if successful, None if failed
    """
    try:
        # Validate file exists and is an image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in ['.png', '.svg', '.jpg', '.jpeg']:
            raise ValueError(f"Unsupported file format: {file_ext}. Only PNG, SVG, and JPG are supported.")
        
        print(f"Parsing recipe from image: {image_path}")
        
        # Parse recipe from image
        recipe = parse_recipe_from_image(image_path)
        
        if dry_run:
            print("=== DRY RUN - Recipe Data ===")
            print(f"Name: {recipe.name}")
            print(f"Ingredients: {recipe.ingredients}")
            print(f"Instructions: {recipe.instructions}")
            print(f"Description: {recipe.description}")
            print(f"Cuisine Type: {recipe.cuisine_type}")
            print(f"Prep Time: {recipe.prep_time} minutes")
            print(f"Cook Time: {recipe.cook_time} minutes")
            print(f"Total Time: {recipe.total_time} minutes")
            print(f"Servings: {recipe.servings}")
            print("=== End Recipe Data ===")
            print(f"Dry run completed for recipe: {recipe.name}")
        else:
            # Store in database
            recipe_id = store_recipe_in_database(recipe)
            print(f"Successfully uploaded recipe: {recipe.name} with ID: {recipe_id}")
        
        return recipe
        
    except Exception as e:
        print(f"Error uploading recipe from image: {e}")
        return None

def upload_recipe_from_image_data(image_data: bytes, filename: str, dry_run: bool = False) -> Optional[Recipe]:
    """
    Upload recipe from image data (useful for web uploads).
    
    Args:
        image_data: Raw image data as bytes
        filename: Original filename
        dry_run: If True, only parse and print the recipe without storing to database
        
    Returns:
        The Recipe object if successful, None if failed
    """
    try:
        # Validate file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in ['.png', '.svg', '.jpg', '.jpeg']:
            raise ValueError(f"Unsupported file format: {file_ext}. Only PNG, SVG, and JPG are supported.")
        
        # Save image data to temporary file
        temp_path = f"temp_upload_{filename}"
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        try:
            # Parse and store recipe
            recipe = upload_recipe_from_image(temp_path, dry_run)
            return recipe
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Error uploading recipe from image data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python upload_recipe.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    recipe = upload_recipe_from_image(image_path, dry_run=True)
    
    if recipe:
        print(f"Recipe parsing completed (dry run mode)")
    else:
        print("Failed to parse recipe")
        sys.exit(1) 