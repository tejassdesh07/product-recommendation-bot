import os
import sqlite3
import json
import subprocess
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionRecommendProductWithOllama(Action):
    
    def name(self) -> Text:
        return "action_recommend_product_with_ollama"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get the user's query from the latest message
        query = tracker.latest_message.get('text')
        print(f"User query: {query}")  # Log user query

        # Retrieve product recommendations from the SQLite DB based on Ollama embeddings
        results = self.search_products_with_ollama(query)

        if results:
            selected_product = results[0][0]  # Choose the top product
            title = selected_product[1]
            description = selected_product[3]
            variants = json.loads(selected_product[10])  # Assuming variants are stored as JSON
            print(f"Selected product: {title}, Description: {description}, Variants: {variants}")  # Log selected product details

            # Check for color and size variants
            colors = {v['color'] for v in variants if 'color' in v}
            sizes = {v['size'] for v in variants if 'size' in v}
            print(f"Available colors: {colors}, Available sizes: {sizes}")  # Log available variants

            # Ask for preferences if variants exist
            if colors:
                color_buttons = [{'title': color, 'payload': f"/choose_color{{'color':'{color}'}}"} for color in colors]
                dispatcher.utter_message(text="Which color would you prefer?", buttons=color_buttons)
                print(f"Color buttons: {color_buttons}")  # Log color buttons

            if sizes:
                size_buttons = [{'title': size, 'payload': f"/choose_size{{'size':'{size}'}}"} for size in sizes]
                dispatcher.utter_message(text="Which size would you prefer?", buttons=size_buttons)
                print(f"Size buttons: {size_buttons}")  # Log size buttons

            # Ask if user wants to add to cart or buy
            dispatcher.utter_message(text=f"Would you like to add {title} to your cart?", 
                                     buttons=[{'title': 'Add to Cart', 'payload': '/add_to_cart'},
                                              {'title': 'Buy Now', 'payload': '/buy_now'}])
            print("Recommendation prompt sent to user.")  # Log recommendation prompt
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find any products matching your query.")
            print("No products found matching the query.")  # Log no product found

        return []

    def cosine_similarity(self, vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a ** 2 for a in vec1) ** 0.5
        norm2 = sum(b ** 2 for b in vec2) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0
        print(f"Cosine similarity calculated: {similarity}")  # Log cosine similarity
        return similarity

    def search_products_with_ollama(self, query):
        print(f"Searching products with query: {query}")  # Log search initiation
        query_embedding = self.get_ollama_embedding(query)

        if query_embedding is None:
            print("No embedding generated for the query.")  # Log embedding failure
            return []  # If embedding is not generated, return no results

        # Connect to SQLite database and retrieve product data
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'products.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM products')
        products = c.fetchall()
        results = []

        for product in products:
            product_vector_str = product[-1]  # Assuming the vector is stored as JSON in the last column
            if product_vector_str and isinstance(product_vector_str, str):
                try:
                    product_vector = json.loads(product_vector_str)  # Load the vector
                    similarity = self.cosine_similarity(query_embedding, product_vector)
                    print(f"Product ID: {product[0]}, Similarity: {similarity}")  # Log product similarity
                except json.JSONDecodeError:
                    print(f"JSON decoding error for product: {product}")  # Log JSON error
                    continue  # Skip this product if JSON is invalid
            else:
                print(f"Skipping product with no vector: {product}")  # Log skipping product
                continue  # Skip this product if the vector is None or empty
            
            results.append((product, similarity))

        # Sort results by similarity score (most similar first)
        results.sort(key=lambda x: x[1], reverse=True)
        conn.close()

        print(f"Top recommendations: {results[:5]}")  # Log top results
        return results[:5]

    def get_ollama_embedding(self, query):
        try:
            # Replace this with the actual path or command to your locally running Ollama
            command = ["ollama", "embed", query]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                embedding = json.loads(result.stdout.decode('utf-8')).get('embedding')
                print(f"Generated embedding: {embedding}")  # Log generated embedding
                return embedding
            else:
                print(f"Ollama error: {result.stderr.decode('utf-8')}")  # Log Ollama error
                return None
        except Exception as e:
            print(f"Error in getting Ollama embedding: {e}")  # Log any exceptions
            return None


if __name__ == "__main__":
    # Simulate a Rasa environment for testing
    class DummyDispatcher:
        def utter_message(self, text, buttons=None):
            print(f"Bot: {text}")
            if buttons:
                print("Buttons:", buttons)

    class DummyTracker:
        def __init__(self, text):
            self.latest_message = {"text": text}

    # Create instances of the dummy classes
    dispatcher = DummyDispatcher()
    tracker = DummyTracker("I'd like to see backpacks")

    # Create an instance of your action class and run it
    action = ActionRecommendProductWithOllama()
    action.run(dispatcher, tracker, {})
