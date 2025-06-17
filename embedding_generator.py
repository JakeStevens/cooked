from typing import List, Optional
import json
import os
import openai
from types import Recipe


class EmbeddingModel:
    """A generic embedding model that uses an OpenAI-compatible API."""

    def __init__(self,
                 model_name: str,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """Create a new EmbeddingModel.

        Args:
            model_name: The name of the embedding model to use.
            api_key: API key for the embedding service. If ``None``, the environment
                variable ``OPENAI_API_KEY`` will be used.
            base_url: Base URL for the OpenAI-compatible endpoint. If ``None``,
                the client will use the default OpenAI API URL.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "An API key must be provided via argument or OPENAI_API_KEY environment variable."
            )

        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text."""
        response = self.client.embeddings.create(model=self.model_name,
                                                  input=text)
        return response.data[0].embedding


class GeminiEmbeddingModel(EmbeddingModel):
    """Embedding model for Gemini, using the OpenAI-compatible API."""

    def __init__(self, api_key: Optional[str] = None):
        """Create a new GeminiEmbeddingModel.

        Args:
            api_key: API key for the Gemini service. If ``None``, the environment
                variable ``GEMINI_API_KEY`` (preferred) or ``OPENAI_API_KEY`` will
                be used.
        """
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv(
            "OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "An API key must be provided via argument or GEMINI_API_KEY/OPENAI_API_KEY environment variables."
            )

        super().__init__(
            model_name="text-embedding-004",
            api_key=resolved_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

def format_recipe_text(recipe: Recipe) -> str:
    """Convert a Recipe object into a single string for embedding generation."""
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
        f"Servings: {recipe.servings or 'N/A'}"
    ]
    return "\n".join(components)

def generate_embedding_vector(recipe: Recipe, model: EmbeddingModel) -> List[float]:
    """Generate an embedding vector for a recipe using the specified model."""
    recipe_text = format_recipe_text(recipe)
    return model.generate_embedding(recipe_text) 