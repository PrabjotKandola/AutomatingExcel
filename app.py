from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        # save the input file
        input_file = request.files['input_file']
        input_path = os.path.join(UPLOAD_FOLDER, 'input.xlsx')
        input_file.save(input_path)

        # save the reference file
        reference_file = request.files['reference_file']
        reference_path = os.path.join(UPLOAD_FOLDER, 'reference.xlsx')
        reference_file.save(reference_path)

        # Load the input and reference data
        input_data = pd.read_excel(input_path)
        reference_data = pd.read_excel(reference_path)

        # Combine all descriptions from reference data
        all_descriptions = reference_data['BrickDefinition_Includes'].dropna().tolist()

        # Create a TF-IDF vectorizer and fit it to all descriptions
        vectorizer = TfidfVectorizer().fit(all_descriptions)

        # For each product, compute its similarity with every description in reference data
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
            input_data.at[index, 'Category or Short Description'] = 'dispensers' # This needs refining based on your data
            input_data.at[index, 'Related IDs'] = ' '.join(map(str, related_ids))
            input_data.at[index, 'Descriptions for the Related IDs'] = ', '.join(related_descriptions)
            input_data.at[index, 'Single ID (possibly a main or primary related ID)'] = related_ids[0]
            input_data.at[index, 'Description for the Single ID'] = related_descriptions[0]

        # Save the results
        output_path = os.path.join(UPLOAD_FOLDER, 'output.xlsx')
        input_data.to_excel(output_path, index=False)

        return 'Files processed. Check the output.xlsx in your uploads directory.'

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
