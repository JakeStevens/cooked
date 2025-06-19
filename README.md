# Recipe Format
Our recipe schema is:

id, name, ingredients, instructions, description, cuisine\_type, prep\_time, cook\_time, total\_time, servings, source

There is also an embedding table which has the schema:

recipe\_id (foreign key on recipe id), embedding\_vector

To get set up:

1) Run create\_recipe\_db.py, this will pull down 200 recipes from eight portions
2) Run generate\_embeddings.py, this will create the embeddings for them
3) Run query\_recipe.ppy, this will verify everything worked
