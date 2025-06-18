import pytest
from cooked.embedding_generator import EmbeddingModel, GeminiEmbeddingModel, format_recipe_text, generate_embedding_vector
from cooked.cooked_types import Recipe


def test_format_recipe_text():
    recipe = Recipe(
        name="Test Recipe",
        description="A tasty test recipe.",
        cuisine_type="Test Cuisine",
        ingredients=["1 cup flour", "2 eggs"],
        instructions=["Mix ingredients.", "Bake for 20 minutes."],
        prep_time=10,
        cook_time=20,
        total_time=30,
        servings=4
    )
    text = format_recipe_text(recipe)
    assert "Test Recipe" in text
    assert "1 cup flour" in text
    assert "Bake for 20 minutes." in text
    assert "Prep Time: 10 minutes" in text
    assert "Servings: 4" in text


def test_generate_embedding_vector_monkeypatch(monkeypatch):
    # Mock EmbeddingModel.generate_embedding to avoid real API calls
    class DummyEmbeddingModel(EmbeddingModel):
        def __init__(self):
            pass
        def generate_embedding(self, text):
            return [0.1, 0.2, 0.3]

    recipe = Recipe(
        name="Test Recipe",
        ingredients=["1 cup flour"],
        instructions=["Mix ingredients."]
    )
    model = DummyEmbeddingModel()
    embedding = generate_embedding_vector(recipe, model)
    assert embedding == [0.1, 0.2, 0.3] 
