from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model and pipeline
model = joblib.load('Model.pkl')
pipeline = joblib.load('pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    try:
        input_data = {
            'Drug target': data['drug_target'],
            'Target Pathway': data['target_response'],
            'n_feature_pos': float(data['n_feature_pos']),
            'n_feature_neg': float(data['n_feature_neg']),
            'log_ic50_mean_pos': float(data['log_ic50_mean_pos']),
            'log_ic50_mean_neg': float(data['log_ic50_mean_neg']),
            'feature_ic50_t_pval': float(data['feature_ic50_t_pval']),
            'feature_delta_mean_ic50': float(data['feature_delta_mean_ic50']),
            'feature_pos_ic50_var': float(data['feature_pos_ic50_var']),
            'feature_neg_ic50_var': float(data['feature_neg_ic50_var']),
            'feature_pval': float(data['feature_pval']),
            'tissue_pval': float(data['tissue_pval']),
            'msi_pval': float(data['msi_pval']),
            'fdr': float(data['fdr']),
        }
    except KeyError as e:
        return jsonify({'error': f'Missing input: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400

    print("Received data:", data)
    print("Processed input data:", input_data)

    try:
        # Ensure the input data is in a DataFrame for preprocessing
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        processed_data = pipeline.transform(input_df)
        print("Processed data shape:", processed_data.shape)  # Add more logging
        print("Processed data:", processed_data)  # Add more logging
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction = float(prediction)  # Correct conversion to float
        return jsonify({'prediction': prediction})
    except Exception as e:
        print("Error occurred:", e)  # Log the specific error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
