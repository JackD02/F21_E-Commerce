import pandas as pd
import numpy as np
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix

def initialize_new_product_factors(new_product_id, products, model, product_id_to_item_index):
    new_product_data = products[products['productID'] == new_product_id].iloc[0]

    # Extract features
    category = new_product_data['category']
    price = new_product_data['price']
    brand = new_product_data['brand']
    colour = new_product_data['colour']
    material = new_product_data['material']

    # Price (Normalize)
    price_mean = products['price'].mean()
    price_std = products['price'].std()
    normalized_price = (price - price_mean) / price_std

    # One-Hot Encoding (hashing)
    brand_hash = hash(brand) % 5
    colour_hash = hash(colour) % 5
    material_hash = hash(material) % 5

    # Combine Features into a Feature Vector
    feature_vector = np.zeros(model.item_factors.shape[1])
    feature_vector[0] = normalized_price
    feature_vector[1] = brand_hash
    feature_vector[2] = colour_hash
    feature_vector[3] = material_hash

    # Find similar products
    similar_products = products[
        (products['category'] == category) &
        (abs(products['price'] - price) < (price_std * 1.5)) &
        (products['brand'] == brand) &
        (products['colour'] == colour) &
        (products['material'] == material)
    ]['productID'].values

    existing_similar_indices = [product_id_to_item_index[pid] for pid in similar_products if pid in product_id_to_item_index]

    if existing_similar_indices:
        average_factor = np.mean(model.item_factors[existing_similar_indices], axis=0)
        new_product_factor = average_factor
    else:
        # If no similar products, use random initialization
        new_product_factor = np.random.normal(size=model.item_factors.shape[1])

    # Adjust the new product factor slightly based on feature vector
    new_product_factor = new_product_factor + feature_vector * 0.05

    model.item_factors = np.vstack([model.item_factors, new_product_factor])
    new_product_index = model.item_factors.shape[0] - 1
    product_id_to_item_index[new_product_id] = new_product_index
    return new_product_index, product_id_to_item_index

# Load data
purchases = pd.read_csv('purchases.csv')
products = pd.read_csv('products.csv')
users = pd.read_csv('users.csv')

purchases['interaction'] = 1
purchases['confidence'] = purchases['returned'].apply(lambda x: 2 if x == 0 else 0.5)

rows = purchases['userID'].astype('category').cat.codes
cols = purchases['productID'].astype('category').cat.codes
data = purchases['interaction']

# Coordinate matrix
interaction_matrix = coo_matrix((data, (rows, cols)))
confidence_matrix = coo_matrix((purchases['confidence'], (rows, cols)))
interaction_matrix_with_confidence = interaction_matrix.multiply(confidence_matrix)

model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
model.fit(interaction_matrix_with_confidence)

# Map from productID to item index in the interaction matrix
product_id_to_item_index = dict(zip(purchases['productID'].astype('category').cat.categories, range(len(purchases['productID'].astype('category').cat.categories))))

# Recommend top 5 products for a user
user_id = 222
user_index = purchases['userID'].astype('category').cat.codes[purchases['userID'] == user_id].values[0]
user_interaction_vector = interaction_matrix_with_confidence.getrow(user_index)
recommended = model.recommend(user_index, user_interaction_vector, N=5)
print(recommended)

# Map recommended indices to actual product IDs
recommended_product_ids = products.iloc[recommended[0]]['productID'].values
print("For User " + str(user_id) + ":")
print("Recommended Product IDs:", recommended_product_ids)

recommended_products_with_scores = list(zip(recommended[0], recommended[1]))
print("Recommended Products with Scores:")
for product_idx, score in recommended_products_with_scores:
    product_id = products.iloc[product_idx]['productID']
    print("Product ID: " + str(product_id) +", Score: " + str(score))

# Recommend new product (productID == 201)
new_product_id = 201
users_percent = 10

if new_product_id not in products['productID'].values:
    print("Product ID " + str(new_product_id) + " not found in products.csv.")
else:
    if new_product_id in product_id_to_item_index:
        new_product_index = product_id_to_item_index[new_product_id]
    else:
        print("Product " + str(new_product_id) + " does not exist in the training data. Initializing factors for the new product.")
        new_product_index, product_id_to_item_index = initialize_new_product_factors(new_product_id, products, model, product_id_to_item_index)

    # Compute the predicted scores for each user-product pair for the new product
    user_scores = model.user_factors.dot(model.item_factors[new_product_index])

    # Sort the users by their predicted score
    sorted_user_indices = np.argsort(user_scores)[::-1]

    num_top_users = int(len(user_scores) * (users_percent/100))
    top_users = sorted_user_indices[:num_top_users]

    recommended_users = purchases.iloc[top_users]['userID'].unique()
    print("Top " + str(users_percent) + "% of users most likely to buy product " + str(new_product_id) + ": " + str(recommended_users))