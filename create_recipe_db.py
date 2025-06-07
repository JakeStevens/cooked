import sqlite3
import random

def generate_random_recipe_data():
    """Generates a dictionary with random recipe data."""
    recipe_number = random.randint(1, 1000)
    title = f"Recipe Title {recipe_number}"

    possible_ingredients = ["Flour", "Sugar", "Eggs", "Milk", "Butter", "Salt", "Pepper", "Onion", "Garlic", "Chicken", "Beef", "Potatoes", "Carrots", "Celery", "Tomatoes", "Cheese"]
    num_ingredients = random.randint(3, 7)
    ingredients = ", ".join(random.sample(possible_ingredients, num_ingredients))

    possible_actions = ["Mix", "Chop", "Saute", "Bake", "Boil", "Stir", "Fry", "Roast", "Grill", "Simmer"]
    possible_subjects = ["ingredients", "vegetables", "meat", "sauce", "mixture", "dough"]
    num_instructions = random.randint(3, 5)
    instructions_list = []
    for i in range(1, num_instructions + 1):
        action = random.choice(possible_actions)
        subject = random.choice(possible_subjects)
        instructions_list.append(f"Step {i}: {action} the {subject}.")
    instructions = " ".join(instructions_list)

    prep_time = random.randint(10, 120)

    return {
        'title': title,
        'ingredients': ingredients,
        'instructions': instructions,
        'prep_time': prep_time
    }

def create_database():
    """Creates and populates the recipes.db SQLite database."""
    db_name = 'recipes.db'
    conn = None  # Initialize conn to None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Drop the table if it already exists
        cursor.execute("DROP TABLE IF EXISTS recipes_table")

        # Create the table
        cursor.execute('''
            CREATE TABLE recipes_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                ingredients TEXT,
                instructions TEXT,
                prep_time INTEGER
            )
        ''')

        # Populate the table with 50 random recipes
        for _ in range(50):
            recipe_data = generate_random_recipe_data()
            cursor.execute('''
                INSERT INTO recipes_table (title, ingredients, instructions, prep_time)
                VALUES (?, ?, ?, ?)
            ''', (recipe_data['title'], recipe_data['ingredients'], recipe_data['instructions'], recipe_data['prep_time']))

        conn.commit()
        print(f"Database '{db_name}' created and populated with 50 recipes successfully.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_database()
