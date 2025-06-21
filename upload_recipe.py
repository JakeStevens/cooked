import sqlite3
import json
import os
import base64
from typing import Dict, Any, Optional
import openai
from PIL import Image
import io

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_recipe_from_image(image_path: str) -> Dict[str, Any]:
    """Parse recipe from image using Gemini Vision API."""
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
        
        parsed_recipe = json.loads(content)
        return parsed_recipe
    except json.JSONDecodeError as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        raise

def store_recipe_in_database(recipe_data: Dict[str, Any], dry_run: bool = False) -> Optional[int]:
    """Store the parsed recipe data in the SQLite database."""
    if dry_run:
        print("=== DRY RUN - Recipe Data ===")
        print(json.dumps(recipe_data, indent=2))
        print("=== End Recipe Data ===")
        return None
    
    db_name = 'recipes.db'
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Convert lists to JSON strings for storage
        ingredients_json = json.dumps(recipe_data['ingredients'])
        instructions_json = json.dumps(recipe_data['instructions'])

        cursor.execute('''
            INSERT INTO recipes_table (
                name, ingredients, instructions, description,
                cuisine_type, prep_time, cook_time, total_time,
                servings, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            recipe_data['name'],
            ingredients_json,
            instructions_json,
            recipe_data['description'],
            recipe_data['cuisine_type'],
            recipe_data['prep_time'],
            recipe_data['cook_time'],
            recipe_data['total_time'],
            recipe_data['servings'],
            recipe_data['source']
        ))
        
        recipe_id = cursor.lastrowid
        conn.commit()
        print(f"Recipe '{recipe_data['name']}' stored with ID: {recipe_id}")
        return recipe_id

    except Exception as e:
        print(f"Error storing recipe in database: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def upload_recipe_from_image(image_path: str, dry_run: bool = False) -> Optional[int]:
    """
    Main function to upload a recipe from an image.
    
    Args:
        image_path: Path to the PNG, SVG, or JPG image file
        dry_run: If True, only parse and print the recipe without storing to database
        
    Returns:
        The recipe ID if successful, None if failed or dry_run=True
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
        recipe_data = parse_recipe_from_image(image_path)
        
        # Store in database (or just print if dry_run)
        recipe_id = store_recipe_in_database(recipe_data, dry_run)
        
        if dry_run:
            print(f"Dry run completed for recipe: {recipe_data['name']}")
        else:
            print(f"Successfully uploaded recipe: {recipe_data['name']}")
        return recipe_id
        
    except Exception as e:
        print(f"Error uploading recipe from image: {e}")
        return None

def upload_recipe_from_image_data(image_data: bytes, filename: str, dry_run: bool = False) -> Optional[int]:
    """
    Upload recipe from image data (useful for web uploads).
    
    Args:
        image_data: Raw image data as bytes
        filename: Original filename
        dry_run: If True, only parse and print the recipe without storing to database
        
    Returns:
        The recipe ID if successful, None if failed or dry_run=True
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
            recipe_id = upload_recipe_from_image(temp_path, dry_run)
            return recipe_id
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
    recipe_id = upload_recipe_from_image(image_path, dry_run=True)
    
    if recipe_id:
        print(f"Recipe uploaded successfully with ID: {recipe_id}")
    else:
        print("Recipe parsing completed (dry run mode)") 