import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the input and reference data
input_data = pd.read_excel('input.xlsx')
reference_data = pd.read_excel('home.xlsx')

# Combine all descriptions from reference data into a single dataframe
all_descriptions = reference_data['BrickDefinition_Includes'].dropna().tolist()

# Create a TF-IDF vectorizer and fit it to all descriptions
vectorizer = TfidfVectorizer().fit(all_descriptions)

# For each product, compute its similarity with every description in reference data and get the top matches
for index, row in input_data.iterrows():
    product_description = row['Product/Item Description']
    product_vector = vectorizer.transform([product_description])
    
    all_vectors = vectorizer.transform(all_descriptions)
    similarities = cosine_similarity(product_vector, all_vectors)
    
    # Get top N matches (let's say top 4 for the "Related IDs")
    top_n_indices = similarities[0].argsort()[-4:][::-1]
    related_ids = reference_data.iloc[top_n_indices]['BrickCode'].tolist()
    related_descriptions = reference_data.iloc[top_n_indices]['BrickDefinition_Includes'].tolist()
    
    # Store the results back into the input dataframe
    input_data.at[index, 'Category or Short Description'] = 'dispensers' # This might need more logic if not all products are dispensers
    input_data.at[index, 'Related IDs'] = ' '.join(map(str, related_ids))
    input_data.at[index, 'Descriptions for the Related IDs'] = ', '.join(related_descriptions)
    input_data.at[index, 'Single ID (possibly a main or primary related ID)'] = related_ids[0]
    input_data.at[index, 'Description for the Single ID'] = related_descriptions[0]

# Save the results
input_data.to_excel('output.xlsx', index=False)
