import unittest
from unittest.mock import patch, MagicMock, call
import json
import sqlite3
import numpy as np

# Assuming query_recipe.py is in the parent directory or accessible via PYTHONPATH
from query_recipe import find_similar_recipes

class TestFindSimilarRecipes(unittest.TestCase):

    def setUp(self):
        # Mock GeminiEmbeddingModel
        self.mock_gemini_model_patch = patch('query_recipe.GeminiEmbeddingModel')
        self.MockGeminiModelClass = self.mock_gemini_model_patch.start()
        self.mock_gemini_instance = self.MockGeminiModelClass.return_value
        self.mock_gemini_instance.generate_embedding.return_value = [0.1] * 768 # Default query embedding

        # Mock sqlite3.connect
        self.mock_sqlite_connect_patch = patch('query_recipe.sqlite3.connect')
        self.mock_connect = self.mock_sqlite_connect_patch.start()
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_connect.return_value = self.mock_conn
        self.mock_conn.cursor.return_value = self.mock_cursor

        # Default dummy recipe data
        self.sample_recipes_details = {
            1: (1, "Spaghetti Carbonara", "Spaghetti, Eggs, Pancetta", "Cook it.", "Pasta"),
            2: (2, "Chicken Stir-fry", "Chicken, Veggies, Soy Sauce", "Stir-fry it.", "Asian"),
            3: (3, "Tomato Soup", "Tomatoes, Onion, Broth", "Simmer it.", "Soup"),
            4: (4, "Pancakes", "Flour, Milk, Eggs", "Flip it.", "Breakfast"),
        }
        self.sample_embeddings = [
            (1, json.dumps([0.1] * 768)), # Perfect match for default query
            (2, json.dumps([0.2] * 768)),
            (3, json.dumps([0.9] * 768)), # Highest similarity if query is [0.1]
            (4, json.dumps([0.3] * 768)),
        ]
        # Scores for query [0.1]*768:
        # Recipe 1 (ID 1, emb [0.1]): sim = 1.0 (approx, due to normalization in func)
        # Recipe 2 (ID 2, emb [0.2]): sim = around 0.999...
        # Recipe 3 (ID 3, emb [0.9]): sim = around 0.999...
        # Recipe 4 (ID 4, emb [0.3]): sim = around 0.999...
        # For distinct similarity, let's adjust embeddings
        self.sample_embeddings_for_ordering = [
            (1, json.dumps(np.random.rand(768).tolist())), # Random
            (2, json.dumps(self.mock_gemini_instance.generate_embedding.return_value)), # Perfect match
            (3, json.dumps( (np.array(self.mock_gemini_instance.generate_embedding.return_value) * 0.5 + np.random.rand(768) * 0.5).tolist() )), # Medium match
            (4, json.dumps(np.random.rand(768).tolist())), # Random
        ]


    def tearDown(self):
        self.mock_gemini_model_patch.stop()
        self.mock_sqlite_connect_patch.stop()

    def _setup_db_mocks(self, embedding_rows, detail_rows_map):
        # detail_rows_map is a dict {recipe_id: (details_tuple)}
        # This helper will configure cursor.fetchall() for multiple calls if needed.

        # First call to fetchall (for embeddings)
        call_configs = [MagicMock()]
        call_configs[0].fetchall.return_value = embedding_rows

        # Subsequent calls to fetchall (for recipe details)
        # This needs to be dynamic based on selected IDs.
        # We'll mock execute and then fetchall based on the query to recipes_table

        original_execute = self.mock_cursor.execute

        def mock_execute_for_details(query_string, params=None):
            original_execute(query_string, params) # Log the call
            if "SELECT recipe_id, name, ingredients, instructions, category FROM recipes_table WHERE recipe_id IN" in query_string:
                # This is the query for recipe details
                # The params are the recipe_ids
                fetched_details = []
                if params: # params is a list of recipe_ids
                    for recipe_id in params:
                        if recipe_id in detail_rows_map:
                            fetched_details.append(detail_rows_map[recipe_id])
                self.mock_cursor.fetchall.return_value = fetched_details

        self.mock_cursor.execute = mock_execute_for_details
        self.mock_cursor.fetchall.side_effect = None # Clear side_effect if any
        self.mock_cursor.fetchall.return_value = embedding_rows # Default for first call


    def test_returns_list_of_dicts_basic(self):
        # Setup to return one embedding, and details for that one recipe
        recipe_id_to_test = 1
        details_for_test = {recipe_id_to_test: self.sample_recipes_details[recipe_id_to_test]}
        embeddings_for_test = [emb for emb in self.sample_embeddings if emb[0] == recipe_id_to_test]

        self._setup_db_mocks(embeddings_for_test, details_for_test)

        result = find_similar_recipes("test query", top_n=1)
        self.assertIsInstance(result, list)
        if result:
            self.assertEqual(len(result), 1) # Ensure we got one result back
            self.assertIsInstance(result[0], dict)
            self.assertEqual(result[0]["recipe_id"], recipe_id_to_test) # Verify correct recipe
        else:
            # This case might occur if the setup is tricky or recipe_id doesn't match.
            # Given the _setup_db_mocks, we expect a result.
            self.fail("Expected at least one result for basic test but got none.")


    def test_returns_correct_number_of_recipes_top_n(self):
        self._setup_db_mocks(self.sample_embeddings, self.sample_recipes_details)
        result = find_similar_recipes("test query", top_n=2)
        self.assertEqual(len(result), 2)

    def test_returns_all_if_fewer_than_top_n(self):
        self._setup_db_mocks(self.sample_embeddings[:2], {k:v for i, (k,v) in enumerate(self.sample_recipes_details.items()) if i < 2})
        result = find_similar_recipes("test query", top_n=3)
        self.assertEqual(len(result), 2)

    def test_results_contain_expected_keys(self):
        recipe_id_to_test = 1
        details_for_test = {recipe_id_to_test: self.sample_recipes_details[recipe_id_to_test]}
        embeddings_for_test = [emb for emb in self.sample_embeddings if emb[0] == recipe_id_to_test]

        self._setup_db_mocks(embeddings_for_test, details_for_test)

        result = find_similar_recipes("test query", top_n=1)
        self.assertTrue(result, "Expected results, but got an empty list.") # Ensure result is not empty
        if result: # Should always be true given the assertion above
            recipe = result[0]
            self.assertEqual(recipe["recipe_id"], recipe_id_to_test)
            self.assertIn("recipe_id", recipe)
            self.assertIn("name", recipe)
            self.assertIn("ingredients", recipe)
            self.assertIn("instructions", recipe)
            self.assertIn("category", recipe)
            self.assertIn("similarity_score", recipe)

    def test_mocked_similarity_and_order(self):
        # Use embeddings that will produce distinct similarities with query [0.1]*768
        # Define a query embedding
        query_embedding = np.array([0.1, 0.2, 0.3] * (768//3), dtype=float).tolist() # Example 768d query vector
        self.mock_gemini_instance.generate_embedding.return_value = query_embedding

        # Create embeddings:
        # Recipe 2: High similarity (e.g., very close to query_embedding)
        # Recipe 3: Medium similarity
        # Recipe 1: Low similarity
        # Recipe 4: Lowest (or just different)
        # Note: cosine_similarity is sensitive to vector direction, not just magnitude differences.

        # Normalized query embedding for easier reasoning if manual calculation was intended
        query_emb_np_normalized = np.array(query_embedding)
        norm = np.linalg.norm(query_emb_np_normalized)
        if norm == 0: norm = 1 # Avoid division by zero, though unlikely for test data
        query_emb_np_normalized = query_emb_np_normalized / norm

        # Embeddings for DB (ensure they are also lists of floats, matching output of json.loads)
        emb_recipe2 = query_emb_np_normalized.tolist() # Highest similarity
        emb_recipe3 = (query_emb_np_normalized * 0.7 + np.random.rand(768) * 0.3)
        emb_recipe3 = (emb_recipe3 / np.linalg.norm(emb_recipe3)).tolist() # Medium
        emb_recipe1 = (query_emb_np_normalized * 0.3 + np.random.rand(768) * 0.7)
        emb_recipe1 = (emb_recipe1 / np.linalg.norm(emb_recipe1)).tolist() # Low
        emb_recipe4 = (np.random.rand(768) - 0.5) * 2 # Different vector
        emb_recipe4 = (emb_recipe4 / np.linalg.norm(emb_recipe4)).tolist() # Lowest

        test_embeddings = [
            (1, json.dumps(emb_recipe1)),
            (2, json.dumps(emb_recipe2)),
            (3, json.dumps(emb_recipe3)),
            (4, json.dumps(emb_recipe4)),
        ]

        # Shuffle to ensure sorting logic is tested, not just insertion order
        import random
        random.shuffle(test_embeddings)

        self._setup_db_mocks(test_embeddings, self.sample_recipes_details)

        result = find_similar_recipes("custom query for ordering", top_n=4)

        self.assertEqual(len(result), 4, "Should return all 4 recipes")

        # Verify order based on expected similarity
        self.assertEqual(result[0]["recipe_id"], 2, "Recipe 2 should be most similar")
        self.assertEqual(result[1]["recipe_id"], 3, "Recipe 3 should be second most similar")
        self.assertEqual(result[2]["recipe_id"], 1, "Recipe 1 should be third most similar")
        self.assertEqual(result[3]["recipe_id"], 4, "Recipe 4 should be least similar")

        # Check that scores are decreasing
        for i in range(len(result) - 1):
            self.assertGreaterEqual(result[i]["similarity_score"], result[i+1]["similarity_score"],
                                    f"Score for {result[i]['name']} not >= {result[i+1]['name']}")


    def test_no_embeddings_in_db(self):
        self._setup_db_mocks([], {}) # No embeddings
        result = find_similar_recipes("any query", top_n=3)
        self.assertEqual(result, [])

    def test_empty_recipes_table_after_finding_ids(self):
        # Embeddings found, but no matching details in recipes_table
        self._setup_db_mocks(self.sample_embeddings[:1], {}) # Provide embedding for ID 1, but no details for ID 1
        result = find_similar_recipes("any query", top_n=1)
        # The current implementation will return details for recipes it *can* find.
        # If an ID from top similarities isn't in recipes_table, it's omitted.
        self.assertEqual(result, [])


    def test_embedding_model_init_failure(self):
        self.MockGeminiModelClass.side_effect = ValueError("API key error")
        # No need to setup DB mocks as it should fail before that
        result = find_similar_recipes("any query", top_n=3)
        self.assertEqual(result, [])
        # Check if an error message was printed (optional, depends on logging strategy)

    def test_embedding_generation_failure(self):
        self.mock_gemini_instance.generate_embedding.side_effect = Exception("Network error")
        # No need to setup DB mocks
        result = find_similar_recipes("any query", top_n=3)
        self.assertEqual(result, [])

    def test_database_connection_error(self):
        self.mock_connect.side_effect = sqlite3.Error("Cannot connect to DB")
        result = find_similar_recipes("any query", top_n=3)
        self.assertEqual(result, [])

    def test_database_query_error_fetch_embeddings(self):
        self._setup_db_mocks([],{}) # Initial setup, will be overridden by side_effect
        self.mock_cursor.execute.side_effect = sqlite3.Error("SQL error fetching embeddings")

        result = find_similar_recipes("any query", top_n=3)
        self.assertEqual(result, [])

    def test_database_query_error_fetch_details(self):
        # First DB call (embeddings) works, second (details) fails
        self.mock_cursor.fetchall.return_value = self.sample_embeddings[:1] # Found one embedding

        def mock_execute_wrapper(query_string, params=None):
            if "SELECT recipe_id, embedding_vector FROM recipe_embeddings_table" in query_string:
                return # This one is fine
            elif "SELECT recipe_id, name, ingredients, instructions, category FROM recipes_table" in query_string:
                raise sqlite3.Error("SQL error fetching details")
            # Default behavior for other execute calls if any

        self.mock_cursor.execute.side_effect = mock_execute_wrapper

        result = find_similar_recipes("any query", top_n=1)
        # The function currently catches this and returns an empty list
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
