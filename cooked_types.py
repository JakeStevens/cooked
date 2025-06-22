from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Recipe:
    id: int
    name: str
    ingredients: List[str]
    instructions: List[str]
    description: Optional[str] = None
    cuisine_type: Optional[str] = None
    prep_time: Optional[int] = None
    cook_time: Optional[int] = None
    total_time: Optional[int] = None
    servings: Optional[int] = None
    source: Optional[str] = None 