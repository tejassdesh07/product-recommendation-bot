import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer

import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer

class ActionRecommendAndSelectProduct(Action):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def name(self) -> Text:
        return "action_recommend_product"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Retrieve the product ID from the user's latest message
        product_id = tracker.get_slot('product_id')
        
        # If a product ID is provided, show details
        if product_id:
            product = self.get_product_by_id(product_id)
            if product:
                self.show_product_details(dispatcher, product)  # Show details directly
            else:
                dispatcher.utter_message(text="The product could not be found.")
        else:
            # Handle user query to search for products
            query = tracker.latest_message.get('text')
            print(f"User query: {query}")
            results = self.search_products(query)
            print(f"Search results: {results}")

            # Show multiple products for user to select from
            self.show_multiple_products(dispatcher, results)

        return []

   
    def show_multiple_products(self, dispatcher: CollectingDispatcher, results: List):
        if not results:
            dispatcher.utter_message(text="No products found.")
            return

        # Iterate over each product to show a preview
        for product, _ in results:
            title = product[1]
            product_id = product[0]  # Assuming product[0] is the unique product ID
            images_data = json.loads(product[11]) if product[11] else []
            image_url = images_data[0]['src'] if images_data else "No image available"
            description = "Your product description here"  # Add logic to get the product description
            price = "Product price here"  # Add logic to get the product price
            sizes = ["Small", "Medium", "Large"]  # Replace with actual sizes
            colors = ["Red", "Blue", "Green"]  # Replace with actual colors

            # Create the button to show more details
            button_text = "Show More"
            button_payload = f"show_details_{product_id}"  # Use an intent instead
           
            message = (
                f"**{title}**\n"
                f"![Image]({image_url})\n"
                f"[{button_text}](payload:{button_payload})"  # Assuming you can use markdown links for buttons
            )

            dispatcher.utter_message(text=message)


    # Handle the Show More button action
    def show_product_details(self, dispatcher: CollectingDispatcher, product_id: str):
        # Fetch the product details based on product_id
        # This part should include logic to get the actual details of the product
        title = "Product Title Here"  # Replace with logic to get the actual title
        description = "Detailed description of the product."  # Replace with actual description
        price = "$99.99"  # Replace with actual price
        sizes = ["Small", "Medium", "Large"]  # Replace with actual sizes
        colors = ["Red", "Blue", "Green"]  # Replace with actual colors
        image_url = "https://example.com/image.jpg"  # Replace with actual image URL
        
        # Construct the detailed message
        message = (
            f"---\n**{title}**\n\n"
            f"{description}\n\n"
            f"**Price:** {price}\n"
            f"![Image]({image_url})\n\n"
            f"Available sizes: {', '.join(sizes) if sizes else 'Not available'}\n"
            f"Available colors: {', '.join(colors) if colors else 'Not available'}\n"
            f"---"
        )


    # def show_product_details(self, dispatcher: CollectingDispatcher, product):
    #     title = product[1]
    #     description = product[3]
    #     price = self.get_product_price(product)

    #     # Extract images
    #     images_data = json.loads(product[11]) if product[11] else []
    #     image_url = images_data[0]['src'] if images_data else "No image available"

    #     variants = product[10]
    #     sizes, colors = self.extract_variants(variants)

    #     # Construct product message
    #     message = (f"---\n**{title}**\n\n"
    #                f"{description}\n\n"
    #                f"**Price:** {price}\n"
    #                f"![Image]({image_url})\n\n"
    #                f"Available sizes: {', '.join(sizes) if sizes else 'Not available'}\n"
    #                f"Available colors: {', '.join(colors) if colors else 'Not available'}\n"
    #                f"---")

    #     # Send the product details in a new message
    #     dispatcher.utter_message(text=message)

    #     # Ask for size and color selection
    #     dispatcher.utter_message(text="What size would you like? Please specify.")
    #     dispatcher.utter_message(text="What color would you like? Please specify.")

    def search_products(self, query):
        cleaned_query = self.clean_input(query)
        query_vector = self.model.encode(cleaned_query).tolist()
        print(f"Query vector: {query_vector}")

        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'products.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute('SELECT * FROM products')
        products = c.fetchall()

        results = []
        for product in products:
            product_vector_str = product[12]
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

    def get_product_by_id(self, product_id):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'products.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute('SELECT * FROM products WHERE id = ?', (product_id,))
        product = c.fetchone()

        conn.close()
        return product

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

    def extract_variants(self, variants):
        sizes, colors = [], []
        if isinstance(variants, str):
            try:
                variants = json.loads(variants)
                sizes = list(set(variant.get('size') for variant in variants if 'size' in variant))
                colors = list(set(variant.get('color') for variant in variants if 'color' in variant))
            except json.JSONDecodeError:
                sizes, colors = [], []
        return sizes, colors



class ActionShowProductDetails(Action):
    def name(self) -> str:
        return "action_provide_product_details"

    def run(self, dispatcher, tracker, domain):
        product_id = tracker.get_slot("product_id")  # Assume you've set a slot for product_id

        # Fetch product details based on the product_id
        product_details = self.get_product_by_id(product_id)  # Implement this function

        if product_details:
            title = product_details['title']
            image_url = product_details['image_url']
            description = product_details['description']
            price = product_details['price']
            
            message = (
                f"**{title}**\n"
                f"![Image]({image_url})\n"
                f"Description: {description}\n"
                f"Price: {price}\n"+
            )
            dispatcher.utter_message(text=message)
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find the product details.")
    def get_product_by_id(self, product_id):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'products.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute('SELECT * FROM products WHERE id = ?', (product_id,))
        product = c.fetchone()

        conn.close()
        return product
        
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

# actions.py
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List

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