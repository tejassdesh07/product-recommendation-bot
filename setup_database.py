import sqlite3
import json
from sentence_transformers import SentenceTransformer

# Initialize the model for generating product vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Database Connection (SQLite)
# -----------------------------
conn = sqlite3.connect('products.db')  # Creates or connects to the SQLite database
cursor = conn.cursor()

# -----------------------------
# Create Table if Not Exists
# -----------------------------
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    title TEXT,
    handle TEXT,
    body_html TEXT,
    published_at TEXT,
    created_at TEXT,
    updated_at TEXT,
    vendor TEXT,
    product_type TEXT,
    price REAL,
    variantjson TEXT,
    imagejson TEXT,
    product_vector TEXT  -- Store as TEXT for JSON encoding
)
''')

# -----------------------------
# Load JSON Data
# -----------------------------
with open('products.json', 'r', encoding='utf-8') as file:
    products_data = json.load(file)

# Check the structure of the loaded data
if isinstance(products_data, dict):
    products = products_data.get('products', [])
    print(f"Loaded products type: {type(products)}")
    print(f"Loaded products length: {len(products)}")

    if isinstance(products, list):
        # -----------------------------
        # Insert Products into Database
        # -----------------------------
        for product in products:
            # Extract variant prices
            variant_prices = [float(variant['price']) for variant in product.get('variants', [])]

            # Determine the base price for the product
            price = min(variant_prices) if variant_prices else None

            # Convert variant and image data to JSON string to store in the SQLite database
            variantjson = json.dumps(product.get('variants', []))
            imagejson = json.dumps(product.get('images', []))

            # Create a comprehensive product description for vectorization
            product_description = json.dumps(product)  # Convert the whole product dictionary to a JSON string

            # Generate the product vector using the model
            product_vector = model.encode([product_description])[0].tolist()  # Convert to list for storage

            # Convert product_vector to a JSON string (since it's a list)
            product_vector_str = json.dumps(product_vector)

            # Insert the product into the products table
            cursor.execute('''
                INSERT INTO products (id, title, handle, body_html, published_at, created_at, updated_at, vendor, product_type, price, variantjson, imagejson, product_vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                product['id'],
                product['title'],
                product['handle'],
                product['body_html'],
                product['published_at'],
                product['created_at'],
                product['updated_at'],
                product['vendor'],
                product['product_type'],
                price,  # Calculated price from variants
                variantjson,  # JSON-encoded variant data
                imagejson,  # JSON-encoded image data
                product_vector_str  # JSON-encoded product vector (as TEXT)
            ))

        # Commit the changes and close the connection
        conn.commit()
        print("Products, variants, images, and product vectors have been successfully saved to the database.")
    else:
        print("Loaded products is not a list. Please check your JSON data.")
else:
    print("Loaded products type:", type(products_data))
    print("Loaded products length: N/A")
    print("Encountered a non-list item: products")

# -----------------------------
# Close Database Connection
# -----------------------------
conn.close()
