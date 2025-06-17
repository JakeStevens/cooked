from dataclasses import dataclass
from typing import Optional

@dataclass
class Recipe:
    id: int
    name: str
    ingredients: list[str]
    instructions: list[str]
    description: Optional[str] = None
    cuisine_type: Optional[str] = None
    prep_time: Optional[int] = None
    cook_time: Optional[int] = None
    total_time: Optional[int] = None
    servings: Optional[int] = None

# Minimal Recipe class for testing if not already present
class Recipe:
    def __init__(self, name, description=None, cuisine_type=None, ingredients=None, instructions=None, prep_time=None, cook_time=None, total_time=None, servings=None):
        self.name = name
        self.description = description
        self.cuisine_type = cuisine_type
        self.ingredients = ingredients or []
        self.instructions = instructions or []
        self.prep_time = prep_time
        self.cook_time = cook_time
        self.total_time = total_time
        self.servings = servings 