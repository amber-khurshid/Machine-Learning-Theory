# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np
# import google.generativeai as genai  # Import for Gemini API

# from tensorflow.keras.models import load_model
# import os

# app = Flask(__name__)

# # Load models
# log_model = pickle.load(open('bytewise_final_project/models/logistic_model.pkl', 'rb'))
# rf_model = pickle.load(open('bytewise_final_project/models/random_forest_model.pkl', 'rb'))
# nn_model = load_model('bytewise_final_project/models/heart_disease_nn.keras', compile=False)

# # Set up Gemini API key securely (set it in your environment variables)
# # GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# genai.configure(api_key="AIzaSyBvjGZbVixrvwoLLImowUHoI6bJsCPFncI")
# model = genai.GenerativeModel('gemini-pro')  # Specify Gemini Pro model

# # Expected number of features for prediction
# expected_number_of_features = 13

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract form data
#         features = [
#             float(request.form['age']),
#             float(request.form['sex']),
#             float(request.form['cp']),
#             float(request.form['trestbps']),
#             float(request.form['chol']),
#             float(request.form['fbs']),
#             float(request.form['restecg']),
#             float(request.form['thalach']),
#             float(request.form['exang']),
#             float(request.form['oldpeak']),
#             float(request.form['slope']),
#             float(request.form['ca']),
#             float(request.form['thal'])

#         ]

#         # Convert features to numpy array
#         features_array = np.array([features])

#         # Make predictions using the models
#         logistic_pred = log_model.predict(features_array)
#         rf_pred = rf_model.predict(features_array)
#         nn_pred = nn_model.predict(features_array)

#         # Gemini API interaction
#         user_query = "How can I improve my health based on my recent heart disease prediction?"
#         # response = model.start_chat(
#         #     history=[
#         #         {"role": "system", "content": "You are a helpful assistant."},
#         #         {"role": "user", "content": user_query}
#         #     ],
         

#         # )
#         # chatgpt_advice = response.text.strip()  # Access response text
#     history = [
#     {"role": "system", "content": "You are a helpful assistant.", "parts": []},
#     {"role": "user", "content": user_query, "parts": []}
#     ]

#     response = model.start_chat(history=history)

#         # Return predictions and Gemini advice
#          return jsonify({
#             'logistic_prediction': int(logistic_pred[0]),
#             'rf_prediction': int(rf_pred[0]),
#             'nn_prediction': int(nn_pred[0][0] > 0.5),
#             'chatgpt_advice': chatgpt_advice
#         })

#      except Exception as e:
    
#     return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import google.generativeai as genai  # Import for Gemini API

from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load models
log_model = pickle.load(open('bytewise_final_project/models/logistic_model.pkl', 'rb'))
rf_model = pickle.load(open('bytewise_final_project/models/random_forest_model.pkl', 'rb'))
nn_model = load_model('bytewise_final_project/models/heart_disease_nn.keras', compile=False)

# Set up Gemini API key securely (set it in your environment variables)
genai.configure(api_key="AIzaSyBvjGZbVixrvwoLLImowUHoI6bJsCPFncI")

# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')  # Specify Gemini Pro model

# Expected number of features for prediction
expected_number_of_features = 13

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        # Convert features to numpy array
        features_array = np.array([features])

        # Make predictions using the models
        logistic_pred = log_model.predict(features_array)
        rf_pred = rf_model.predict(features_array)
        nn_pred = nn_model.predict(features_array)

        # Gemini API interaction
        user_query = "How can I improve my health based on my recent heart disease prediction?"
        history = [
    {"role": "system", "content": "You are a helpful assistant.", "parts": [
        {"text": "This is the actual content."}
    ]},
    {"role": "user", "content": user_query, "parts": [
        {"text": "My question is..."}
    ]}
    ]

        response = model.start_chat(history=history)
        chatgpt_advice = response.text.strip()  # Access response text

        # Return predictions and Gemini advice
        return jsonify({
            'logistic_prediction': int(logistic_pred[0]),
            'rf_prediction': int(rf_pred[0]),
            'nn_prediction': int(nn_pred[0][0] > 0.5),
            'chatgpt_advice': chatgpt_advice
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



# # from flask import Flask, request, jsonify, render_template
# # import pickle
# # import numpy as np
# # import google.generativeai as genai  # Import for Gemini API

# # from tensorflow.keras.models import load_model
# # import os

# # app = Flask(__name__)

# # # Load models
# # log_model = pickle.load(open('bytewise_final_project/models/logistic_model.pkl', 'rb'))
# # rf_model = pickle.load(open('bytewise_final_project/models/random_forest_model.pkl', 'rb'))
# # nn_model = load_model('bytewise_final_project/models/heart_disease_nn.keras', compile=False)

# # # Set up Gemini API key securely (set it in your environment variables)
# # genai.configure(api_key=os.getenv('GEMINI_API_KEY'))  # Configure API key

# # # Specify Gemini model (make sure this is correct)
# # model_name = 'models/chat-bison-001'  # or use 'gemini-pro' if that's correct

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         # Extract form data
# #         features = [
# #             float(request.form['age']),
# #             float(request.form['sex']),
# #             float(request.form['cp']),
# #             float(request.form['trestbps']),
# #             float(request.form['chol']),
# #             float(request.form['fbs']),
# #             float(request.form['restecg']),
# #             float(request.form['thalach']),
# #             float(request.form['exang']),
# #             float(request.form['oldpeak']),
# #             float(request.form['slope']),
# #             float(request.form['ca']),
# #             float(request.form['thal'])
# #         ]

# #         # Convert features to numpy array
# #         features_array = np.array([features])

# #         # Make predictions using the models
# #         logistic_pred = log_model.predict(features_array)
# #         rf_pred = rf_model.predict(features_array)
# #         nn_pred = nn_model.predict(features_array)

# #         # Gemini API interaction
# #         user_query = "How can I improve my health based on my recent heart disease prediction?"
        
# #         response = genai.ChatSession(
# #             model=model_name,
# #             history = [{"role": "system", "content": "You are a helpful assistant.", "parts": [
# #         {"text": "This is the actual content."}]},{"role": "user", "Content": user_query, "parts": [{"text": "My question is..."}
# #     ]}]
# #             # temperature=0.7  # Optional parameter
# #             )
# #         chatgpt_advice = response['choices'][0]['message']['Content'].strip()  # Access response text

# #         # Return predictions and Gemini advice
# #         return jsonify({
# #             'logistic_prediction': int(logistic_pred[0]),
# #             'rf_prediction': int(rf_pred[0]),
# #             'nn_prediction': int(nn_pred[0][0] > 0.5),
# #             'chatgpt_advice': chatgpt_advice
# #         })

# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template, redirect, url_for
# import pickle
# import numpy as np
# import google.generativeai as genai  # Import for Gemini API
# from tensorflow.keras.models import load_model
# import os

# app = Flask(__name__)

# # Load models
# log_model = pickle.load(open('bytewise_final_project/models/logistic_model.pkl', 'rb'))
# rf_model = pickle.load(open('bytewise_final_project/models/random_forest_model.pkl', 'rb'))
# nn_model = load_model('bytewise_final_project/models/heart_disease_nn.keras', compile=False)

# # Set up Gemini API key securely (set it in your environment variables)
# genai.configure(api_key="AIzaSyBvjGZbVixrvwoLLImowUHoI6bJsCPFncI")
# # # model = genai.GenerativeModel('gemini-pro') 
# # genai.configure(api_key=os.getenv('GEMINI_API_KEY'))  # Configure API key
# model_name = 'models/chat-bison-001'  # or use 'gemini-pro' if that's correct

# # Expected number of features for prediction
# expected_number_of_features = 13

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract form data
#         features = [
#             float(request.form['age']),
#             float(request.form['sex']),
#             float(request.form['cp']),
#             float(request.form['trestbps']),
#             float(request.form['chol']),
#             float(request.form['fbs']),
#             float(request.form['restecg']),
#             float(request.form['thalach']),
#             float(request.form['exang']),
#             float(request.form['oldpeak']),
#             float(request.form['slope']),
#             float(request.form['ca']),
#             float(request.form['thal'])
#         ]

#         # Convert features to numpy array
#         features_array = np.array([features])

#         # Make predictions using the models
#         logistic_pred = log_model.predict(features_array)
#         rf_pred = rf_model.predict(features_array)
#         nn_pred = nn_model.predict(features_array)

#         # Store predictions in session or pass as parameters for redirect
#         return redirect(url_for('chatbot', 
#                                 logistic_pred=int(logistic_pred[0]), 
#                                 rf_pred=int(rf_pred[0]), 
#                                 nn_pred=int(nn_pred[0][0] > 0.5)))
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/chatbot')
# def chatbot():
#     logistic_pred = request.args.get('logistic_pred')
#     rf_pred = request.args.get('rf_pred')
#     nn_pred = request.args.get('nn_pred')

#     user_query = f"How can I improve my health based on my recent heart disease prediction? Logistic: {logistic_pred}, RF: {rf_pred}, NN: {nn_pred}"
    
#     response = genai.chat(
#         model=model_name,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": user_query}
#         ],
#         temperature=0.7  # Optional parameter
#     )
#     chatgpt_advice = response['choices'][0]['message']['content'].strip()  # Access response text

#     return render_template('chatbot.html', advice=chatgpt_advice)

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, request, jsonify, render_template, redirect, url_for

# import pickle
# import numpy as np
# import google.generativeai as genai  # Import for Gemini API
# from tensorflow.keras.models import load_model
# import os

# app = Flask(__name__)

# # Load models
# log_model = pickle.load(open('bytewise_final_project/models/logistic_model.pkl', 'rb'))
# rf_model = pickle.load(open('bytewise_final_project/models/random_forest_model.pkl', 'rb'))
# nn_model = load_model('bytewise_final_project/models/heart_disease_nn.keras', compile=False)

# # Set up Gemini API key securely (set it in your environment variables)
# genai.configure(api_key=os.getenv('GEMINI_API_KEY'))  # Configure API key
# model_name = 'models/chat-bison-001'  # or use 'gemini-pro' if that's correct

# # Expected number of features for prediction
# expected_number_of_features = 13

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract form data
#         features = [
#             float(request.form['age']),
#             float(request.form['sex']),
#             float(request.form['cp']),
#             float(request.form['trestbps']),
#             float(request.form['chol']),
#             float(request.form['fbs']),
#             float(request.form['restecg']),
#             float(request.form['thalach']),
#             float(request.form['exang']),
#             float(request.form['oldpeak']),
#             float(request.form['slope']),
#             float(request.form['ca']),
#             float(request.form['thal'])
#         ]

#         # Convert features to numpy array
#         features_array = np.array([features])

#         # Make predictions using the models
#         logistic_pred = log_model.predict(features_array)
#         rf_pred = rf_model.predict(features_array)
#         nn_pred = nn_model.predict(features_array)

#         # Store predictions in session or pass as parameters for redirect
#         return redirect(url_for('chatbot', 
#                                 logistic_pred=int(logistic_pred[0]), 
#                                 rf_pred=int(rf_pred[0]), 
#                                 nn_pred=int(nn_pred[0][0] > 0.5)))
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/chatbot')
# def chatbot():
#     try:
#         logistic_pred = request.args.get('logistic_pred')
#         rf_pred = request.args.get('rf_pred')
#         nn_pred = request.args.get('nn_pred')

#         user_query = f"How can I improve my health based on my recent heart disease prediction? Logistic: {logistic_pred}, RF: {rf_pred}, NN: {nn_pred}"
        
#         # Fetch chatbot response
#         response = genai.chat(
#             model=model_name,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": user_query}
#             ],
#             temperature=0.7  # Optional parameter
#         )
        
#         # Ensure response contains expected data
#         chatgpt_advice = response['choices'][0]['message']['content'].strip()  # Access response text

#         return render_template('chatbot.html', advice=chatgpt_advice)

#     except Exception as e:
#         # Log exception and return error message
#         print(f"Error in chatbot route: {e}")
#         return render_template('chatbot.html', advice="An error occurred. Please try again later.")

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify, render_template, redirect, url_for
# from flask_cors import CORS  # Import CORS
# import pickle
# import numpy as np
# import google.generativeai as genai
# from tensorflow.keras.models import load_model
# import os

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load models
# log_model = pickle.load(open('bytewise_final_project/models/logistic_model.pkl', 'rb'))
# rf_model = pickle.load(open('bytewise_final_project/models/random_forest_model.pkl', 'rb'))
# nn_model = load_model('bytewise_final_project/models/heart_disease_nn.keras', compile=False)


# # Set up Gemini API key securely
# genai.configure(api_key="AIzaSyBvjGZbVixrvwoLLImowUHoI6bJsCPFncI")
# model_name = 'models/chat-bison-001'

# # Expected number of features for prediction
# expected_number_of_features = 13

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         features = [
#             float(request.form['age']),
#             float(request.form['sex']),
#             float(request.form['cp']),
#             float(request.form['trestbps']),
#             float(request.form['chol']),
#             float(request.form['fbs']),
#             float(request.form['restecg']),
#             float(request.form['thalach']),
#             float(request.form['exang']),
#             float(request.form['oldpeak']),
#             float(request.form['slope']),
#             float(request.form['ca']),
#             float(request.form['thal'])
#         ]

#         features_array = np.array([features])
#         logistic_pred = log_model.predict(features_array)
#         rf_pred = rf_model.predict(features_array)
#         nn_pred = nn_model.predict(features_array)

#         return redirect(url_for('chatbot', 
#                                 logistic_pred=int(logistic_pred[0]), 
#                                 rf_pred=int(rf_pred[0]), 
#                                 nn_pred=int(nn_pred[0][0] > 0.5)))
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/chatbot')
# def chatbot():
#     logistic_pred = request.args.get('logistic_pred')
#     rf_pred = request.args.get('rf_pred')
#     nn_pred = request.args.get('nn_pred')

#     user_query = f"How can I improve my health based on my recent heart disease prediction? Logistic: {logistic_pred}, RF: {rf_pred}, NN: {nn_pred}"
    
#     response = genai.chat(
#         model=model_name,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": user_query}
#         ],
#         temperature=0.7
#     )
#     chatgpt_advice = response['choices'][0]['message']['content'].strip()

#     return render_template('chatbot.html', advice=chatgpt_advice)

# if __name__ == '__main__':
#     app.run(debug=True)
