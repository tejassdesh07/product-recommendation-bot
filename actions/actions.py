

import os
import json
import sqlite3
import re
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer


class ActionRecommendProduct(Action):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def name(self) -> Text:
        return "action_recommend_product"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        query = tracker.latest_message.get('text')
        print(f"User query: {query}")  # Debug output

        results = self.search_products(query)
        print(f"Search results: {results}")  # Debug output

        filtered_results = [result for result in results if result[1] > 0.2]
        if len(filtered_results) == 1:
            product = filtered_results[0][0]
            self.show_product_details(dispatcher, product)
        else:
            self.show_multiple_products(dispatcher, filtered_results)

        return []

    def show_product_details(self, dispatcher: CollectingDispatcher, product):
        title = product[1]
        description = product[3]
        price = self.get_product_price(product)

        images_data = json.loads(product[11]) if product[11] else []
        image_url = images_data[0]['src'] if images_data else "No image available"

        variants = product[10]
        sizes, colors = self.extract_variants(variants)

        message = (f"**{title}**\n\n"
                   f"{description}\n\n"
                   f"**Price:** {price}\n"
                   f"![Image]({image_url})\n\n"
                   f"Available sizes: {', '.join(sizes) if sizes else 'Not available'}\n"
                   f"Available colors: {', '.join(colors) if colors else 'Not available'}")

        dispatcher.utter_message(text=message)

        buttons = [
            {"title": "Select Size", "payload": "/action_ask_for_size"},
            {"title": "Select Color", "payload": "/action_ask_for_color"},
            {"title": "Add to Cart", "payload": "/action_add_to_cart"},
            {"title": "Buy Now", "payload": "/action_buy_now"}
        ]
        dispatcher.utter_message(text="How would you like to proceed?", buttons=buttons)

    def show_multiple_products(self, dispatcher: CollectingDispatcher, results: List):
        if not results:
            dispatcher.utter_message(text="No products found.")
            return

        button_array = []
        product_messages = []

        for product, similarity in results:
            product_id = product[0]
            title = product[1]
            images_data = json.loads(product[11]) if product[11] else []
            image_url = images_data[0]['src'] if images_data else "No image available"

            product_message = f"**{title}**\n![Image]({image_url})"
            product_messages.append(product_message)

            button = {
                "title": title,
                "payload": f'/action_view_product_details'
            }
            button_array.append(button)

        for message in product_messages:
            dispatcher.utter_message(text=message)

        dispatcher.utter_message(text="Would you like more details?", buttons=button_array)

        more_buttons = [
            {"title": "See More Products", "payload": "/action_see_more_products"},
            {"title": "Start Over", "payload": "/action_start_over"},
            {"title": "Add to Cart", "payload": "/action_add_to_cart"}
        ]
        dispatcher.utter_message(text="What would you like to do next?", buttons=more_buttons)

    
    def extract_variants(self, variants):
        sizes, colors = [], []
        if isinstance(variants, str):
            try:
                variants = json.loads(variants)
                sizes = list(set(variant.get('size') for variant in variants if 'size' in variant))
                colors = list(set(variant.get('color') for variant in variants if 'color' in variant))
            except json.JSONDecodeError:
                pass  # Handle invalid JSON
        return sizes, colors

    def get_product_price(self, product):
        variants = product[10]
        price_display = "Price not available"
        if isinstance(variants, str):
            try:
                variants = json.loads(variants)
                price = variants[0].get('price') if variants else None
                price_display = f"${price}" if price else "Price not available"
            except json.JSONDecodeError:
                pass
        return price_display

    def search_products(self, query):
        cleaned_query = self.clean_input(query)
        query_vector = self.model.encode(cleaned_query).tolist()
        print(f"Query vector: {query_vector}")  # Debug output

        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'products.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM products')
        products = c.fetchall()

        results = []
        for product in products:
            product_vector_str = product[-1]
            if product_vector_str and isinstance(product_vector_str, str):
                try:
                    product_vector = json.loads(product_vector_str)
                    similarity = self.cosine_similarity(query_vector, product_vector)
                    results.append((product, similarity))
                except json.JSONDecodeError:
                    continue

        results.sort(key=lambda x: x[1], reverse=True)
        conn.close()
        return results[:5]

    def clean_input(self, text: str) -> str:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text

    def cosine_similarity(self, vec1, vec2):
        return sum(a * b for a, b in zip(vec1, vec2)) / (self.norm(vec1) * self.norm(vec2))

    def norm(self, vec):
        return sum(x ** 2 for x in vec) ** 0.5


class ActionViewProduct(Action):
    def name(self) -> Text:
        return "action_view_product_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        product_id = 7661300154438
        print(f"Fetching product details for ID: {product_id}")  # Debug output
        product = self.fetch_product_by_id(product_id)

        if product:
            self.show_product_details(dispatcher, product)
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find that product.")

        return []

    def fetch_product_by_id(self, product_id):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'products.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM products WHERE id = ?', (product_id,))
        product = c.fetchone()
        conn.close()
        return product

    def show_product_details(self, dispatcher: CollectingDispatcher, product):
        title = product[1]
        description = product[3]
        price = self.get_product_price(product)

        images_data = json.loads(product[11]) if product[11] else []
        image_url = images_data[0]['src'] if images_data else "No image available"

        variants = product[10]
        sizes, colors = self.extract_variants(variants)

        message = (f"**{title}**\n\n"
                   f"{description}\n\n"
                   f"**Price:** {price}\n"
                   f"![Image]({image_url})\n\n"
                   f"Available sizes: {', '.join(sizes) if sizes else 'Not available'}\n"
                   f"Available colors: {', '.join(colors) if colors else 'Not available'}")

        dispatcher.utter_message(text=message)

        buttons = [
            {"title": "Select Size", "payload": "/action_ask_for_size"},
            {"title": "Select Color", "payload": "/action_ask_for_color"},
            {"title": "Add to Cart", "payload": "/action_add_to_cart"},
            {"title": "Buy Now", "payload": "/action_buy_now"}
        ]
        dispatcher.utter_message(text="How would you like to proceed?", buttons=buttons)

    def get_product_price(self, product):
        variants = product[10]
        price_display = "Price not available"
        if isinstance(variants, str):
            try:
                variants = json.loads(variants)
                price = variants[0].get('price') if variants else None
                price_display = f"${price}" if price else "Price not available"
            except json.JSONDecodeError:
                pass
        return price_display




class ActionAskForSize(Action):
    def name(self) -> Text:
        return "action_ask_for_size"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Please select your preferred size.")
        return []

class ActionAskForColor(Action):
    def name(self) -> Text:
        return "action_ask_for_color"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Please select your preferred color.")
        return []

class ActionAddToCart(Action):
    def name(self) -> Text:
        return "action_add_to_cart"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="The product has been added to your cart. Would you like to view your cart or continue shopping?")
        return []

class ActionBuyNow(Action):
    def name(self) -> Text:
        return "action_buy_now"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="You are being redirected to the checkout page.")
        return []

class ActionAskClarification(Action):
    def name(self) -> Text:
        return "action_ask_clarification"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="Could you please clarify what you mean?")
        print("Clarification requested.")  # Debugging: Print clarification request
        return []

class ActionProvideProductDetails(Action):
    def name(self) -> Text:
        return "action_provide_product_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        product_details = "Details about the selected product."
        print(f"Product details: {product_details}")  # Debugging: Print product details
        dispatcher.utter_message(text=product_details)
        return []

class ActionAskForPreferences(Action):
    def name(self) -> Text:
        return "action_ask_for_preferences"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="What specific features or qualities are you looking for?")
        print("Asking for user preferences.")  # Debugging: Print preference request
        return []



class ActionNotifyOutOfStock(Action):
    def name(self) -> Text:
        return "action_notify_out_of_stock"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Example action logic
        dispatcher.utter_message(text="Sorry, the product is out of stock.")
        return []

class ActionSuggestAlternatives(Action):
    def name(self) -> Text:
        return "action_suggest_alternatives"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Example action logic
        dispatcher.utter_message(text="Here are some alternative products you might like.")
        return []

class ActionSuggestRelatedProducts(Action):
    def name(self) -> Text:
        return "action_suggest_related_products"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Example action logic
        dispatcher.utter_message(text="Here are some related products based on your selection.")
        return []

class ActionConfirmCart(Action):
    def name(self) -> Text:
        return "action_confirm_cart"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Example action logic
        dispatcher.utter_message(text="Your cart has been confirmed.")
        return []

class ActionProvideCartSummary(Action):
    def name(self) -> Text:
        return "action_provide_cart_summary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Example action logic
        dispatcher.utter_message(text="Here is the summary of your cart.")
        return []

class ActionCheckoutCart(Action):
    def name(self) -> Text:
        return "action_checkout_cart"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Example action logic
        dispatcher.utter_message(text="Your cart has been checked out.")
        return []

if __name__ == "__main__":
    dispatcher = CollectingDispatcher()
    tracker = Tracker(
        sender_id="test",
        slots={},
        latest_message={"text": "Looking for a red shirt"},
        latest_action_name="action_listen",
        events=[],  # Required: Provide an empty list for events
        paused=False,  # Required: Provide False for paused
        active_loop=None,
        followup_action=None,
    )

    action = ActionRecommendProduct()
    action.run(dispatcher, tracker, {})