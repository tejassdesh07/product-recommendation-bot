import json
import psycopg2
from sentence_transformers import SentenceTransformer

# Initialize the model for generating product vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Database Connection
# -----------------------------
conn = psycopg2.connect(
    dbname='rasa_db',
    user='postgres',
    password='12345',
    host='localhost',
    port='5432'
)
cursor = conn.cursor()

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

            # Insert the product into the products table
            cursor.execute('''
            INSERT INTO products (id, title, handle, body_html, published_at, created_at, updated_at, vendor, product_type, price)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                price
            ))

            # -----------------------------
            # Insert Variants into Database
            # -----------------------------
            for variant in product.get('variants', []):
                cursor.execute('''
                INSERT INTO variants (id, product_id, title, sku, price, requires_shipping, taxable, available, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    variant['id'],
                    product['id'],
                    variant['title'],
                    variant['sku'],
                    float(variant['price']),
                    variant['requires_shipping'],
                    variant['taxable'],
                    variant['available'],
                    variant['created_at'],
                    variant['updated_at']
                ))

            # -----------------------------
            # Insert Images into Database
            # -----------------------------
            for image in product.get('images', []):
                cursor.execute('''
                INSERT INTO images (id, product_id, src, width, height, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (
                    image['id'],
                    product['id'],
                    image['src'],
                    image['width'],
                    image['height'],
                    image['created_at'],
                    image['updated_at']
                ))

        # Commit the changes and close the connection
        conn.commit()
        print("Products, variants, and images have been successfully saved to the database.")
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
