#!/usr/bin/env python3
"""
ResQRelief: Disaster Management System 
(FINAL DEPLOYMENT VERSION)
"""

import pandas as pd
import numpy as np
import re
import joblib
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (Needed for tokenization/NLP)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# The main Flask application instance (Named 'app' for Gunicorn)
app = Flask(__name__) 

class ResQReliefSystem:
    def __init__(self):
        self.flood_model = None
        self.scaler = None
        self.messages_model = None
        self.flood_features = None
        self.category_names = [
            'related', 'request', 'offer', 'aid_related', 'medical_help',
            'medical_products', 'search_and_rescue', 'security', 'military',
            'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
            'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
            'transport', 'buildings', 'electricity', 'tools', 'hospitals',
            'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
            'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'
        ]
        self.setup_models()

    def setup_models(self):
        print("🔄 Training models...")
        self.train_models()

    def train_models(self):
        # --- FLOOD MODEL TRAINING (Using Sample Data from Original Code) ---
        np.random.seed(42)
        n_samples = 1000

        flood_data = {
            'MonsoonIntensity': np.random.normal(0.5, 0.2, n_samples),
            'TopographyDrainage': np.random.uniform(0, 1, n_samples),
            'RiverManagement': np.random.uniform(0, 1, n_samples),
            'Deforestation': np.random.uniform(0, 1, n_samples),
            'Urbanization': np.random.uniform(0, 1, n_samples),
            'ClimateChange': np.random.uniform(0, 1, n_samples),
            'DamsQuality': np.random.uniform(0, 1, n_samples),
            'Siltation': np.random.uniform(0, 1, n_samples),
            'AgriculturalPractices': np.random.uniform(0, 1, n_samples),
            'Encroachments': np.random.uniform(0, 1, n_samples),
            'IneffectiveDisasterPreparedness': np.random.uniform(0, 1, n_samples),
            'DrainageSystems': np.random.uniform(0, 1, n_samples),
            'CoastalVulnerability': np.random.uniform(0, 1, n_samples),
            'Landslides': np.random.uniform(0, 1, n_samples),
            'Watersheds': np.random.uniform(0, 1, n_samples),
            'DeterioratingInfrastructure': np.random.uniform(0, 1, n_samples),
            'PopulationScore': np.random.uniform(0, 1, n_samples),
            'WetlandLoss': np.random.uniform(0, 1, n_samples),
            'InadequatePlanning': np.random.uniform(0, 1, n_samples),
            'PoliticalFactors': np.random.uniform(0, 1, n_samples)
        }

        df_flood = pd.DataFrame(flood_data)

        # Create realistic flood probability target
        flood_prob = (
            0.3 * df_flood['MonsoonIntensity'] +
            0.2 * df_flood['Deforestation'] +
            0.2 * df_flood['Urbanization'] +
            0.1 * df_flood['ClimateChange'] +
            0.1 * (1 - df_flood['RiverManagement']) +
            0.1 * df_flood['IneffectiveDisasterPreparedness'] +
            np.random.normal(0, 0.1, n_samples)
        )

        flood_prob = np.clip(flood_prob, 0, 1)
        df_flood['FloodProbability'] = pd.cut(
            flood_prob,
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(int)

        # Train flood model
        X_flood = df_flood.drop('FloodProbability', axis=1)
        y_flood = df_flood['FloodProbability']

        X_train, X_test, y_train, y_test = train_test_split(
            X_flood, y_flood, test_size=0.2, random_state=42, stratify=y_flood
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.flood_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.flood_model.fit(X_train_scaled, y_train)
        self.flood_features = list(X_flood.columns)

        # --- MESSAGE CLASSIFICATION TRAINING (Using Sample Data from Original Code) ---
        sample_messages = [
            "We need water and food supplies urgently",
            "Medical help required for injured people",
            "Shelter needed for homeless families",
            "Search and rescue teams needed in the area",
            "Electricity infrastructure damaged by storm",
            "Roads blocked due to flooding",
            "Hospital running out of medical supplies",
            "Children separated from families need help",
            "Fire spreading in residential area",
            "Earthquake damaged multiple buildings"
        ]

        np.random.seed(42)
        y_sample = np.random.randint(0, 2, (len(sample_messages), len(self.category_names)))

        # Add logical connections
        for i, msg in enumerate(sample_messages):
            if 'water' in msg or 'food' in msg:
                y_sample[i][self.category_names.index('water')] = 1
                y_sample[i][self.category_names.index('food')] = 1
                y_sample[i][self.category_names.index('aid_related')] = 1
            if 'medical' in msg:
                y_sample[i][self.category_names.index('medical_help')] = 1
            if 'shelter' in msg:
                y_sample[i][self.category_names.index('shelter')] = 1
            if 'rescue' in msg:
                y_sample[i][self.category_names.index('search_and_rescue')] = 1
            if 'fire' in msg:
                y_sample[i][self.category_names.index('fire')] = 1
            if 'earthquake' in msg:
                y_sample[i][self.category_names.index('earthquake')] = 1

        def tokenize(text):
            text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
            tokens = text.split()
            lemmatizer = WordNetLemmatizer()
            try:
                stop_words = set(stopwords.words("english"))
            except:
                stop_words = set()
            clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
            return clean_tokens

        self.messages_model = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize, max_features=1000)),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42)))
        ])

        self.messages_model.fit(sample_messages, y_sample)
        print("✅ Models trained successfully!")

    def predict_flood_risk(self, features):
        try:
            feature_dict = dict(zip(self.flood_features, features))
            input_df = pd.DataFrame([feature_dict])
            input_scaled = self.scaler.transform(input_df)

            prediction = self.flood_model.predict(input_scaled)[0]
            probability = self.flood_model.predict_proba(input_scaled)[0]

            risk_levels = ['Low', 'Medium', 'High', 'Very High']
            risk_level = risk_levels[prediction]
            confidence = max(probability) * 100

            return {
                'risk_level': risk_level,
                'confidence': round(confidence, 2),
                'prediction': int(prediction),
                'probabilities': {
                    'Low': round(probability[0] * 100, 2),
                    'Medium': round(probability[1] * 100, 2),
                    'High': round(probability[2] * 100, 2),
                    'Very High': round(probability[3] * 100, 2)
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def classify_message(self, message):
        try:
            prediction = self.messages_model.predict([message])[0]
            predicted_categories = [
                self.category_names[i] for i, pred in enumerate(prediction) if pred == 1
            ]

            return {
                'predicted_categories': predicted_categories,
                'total_categories': len(predicted_categories),
                'message': message
            }
        except Exception as e:
            return {'error': str(e)}

# Initialize system. This creates the app instance and trains the models when Gunicorn starts.
system = ResQReliefSystem()

# --- WEB ROUTES ---

@app.route('/')
def index():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>ResQRelief - Disaster Management System</title>
    <style>
        /* BASE THEME - DARK MODE */
        :root {
            --bg-start: #0f2027; /* Dark Teal Start */
            --bg-end: #203a43;   /* Dark Blue End */
            --text-color: white;
            --card-bg: rgba(255,255,255,0.1);
            --btn-bg: #1abc9c; /* Bright Teal Button */
            --btn-hover-bg: #16a085;
            --shadow-color: rgba(0,0,0,0.4);
            --toggle-text: yellow;
        }
        
        /* LIGHT MODE OVERRIDES */
        body.light-theme {
            --bg-start: #e0f2f1; /* Light mint/teal */
            --bg-end: #b2dfdb;   /* Slightly darker mint */
            --text-color: #333;
            --card-bg: rgba(0,0,0,0.05);
            --btn-bg: #3498db; /* Blue button */
            --btn-hover-bg: #2980b9;
            --shadow-color: rgba(0,0,0,0.1);
            --toggle-text: #333;
        }

        body { 
            font-family: Arial, sans-serif; 
            margin: 40px; 
            background: linear-gradient(135deg, var(--bg-start) 0%, var(--bg-end) 100%); 
            color: var(--text-color); 
            min-height: 100vh;
            transition: background 0.5s, color 0.5s;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            text-align: center; 
            position: relative; /* For toggle button */
        }
        .card { 
            background: var(--card-bg); 
            padding: 30px; 
            margin: 20px 0; 
            border-radius: 10px; 
            backdrop-filter: blur(5px); 
            box-shadow: 0 4px 15px var(--shadow-color);
            transition: background 0.5s, box-shadow 0.5s;
        }
        .btn { 
            background: var(--btn-bg); 
            color: white; 
            padding: 15px 30px; 
            text-decoration: none; 
            border-radius: 5px; 
            margin: 10px; 
            display: inline-block; 
            transition: all 0.3s;
            border: none;
            cursor: pointer;
        }
        .btn:hover { 
            background: var(--btn-hover-bg); 
            transform: translateY(-2px); 
        }
        h1 { 
            font-size: 3em; 
            margin-bottom: 20px; 
            text-shadow: 2px 2px 4px var(--shadow-color); 
        }
        .feature { 
            display: inline-block; 
            margin: 20px; 
        }
        .theme-toggle-container {
            position: absolute;
            top: 0;
            right: 0;
        }
        .theme-toggle-btn {
            background: none;
            color: var(--toggle-text);
            border: 2px solid var(--toggle-text);
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }
        .theme-toggle-btn:hover {
            background: var(--toggle-text);
            color: var(--bg-start);
        }
    </style>
</head>
<body class="dark-theme">
    <div class="container">
        <div class="theme-toggle-container">
            <button id="theme-toggle" class="theme-toggle-btn">☀️ Light Mode</button>
        </div>
        <h1>🛡️ ResQRelief</h1>
        <p style="font-size: 1.2em;">Integrated Disaster Impact Prediction and Response Management System</p>

        <div class="card">
            <h2>🌊 Flood Risk Prediction</h2>
            <p>Analyze environmental factors and predict flood probability using machine learning</p>
            <a href="/flood-prediction" class="btn">Predict Flood Risk</a>
        </div>

        <div class="card">
            <h2>💬 Message Classification</h2>
            <p>Automatically classify emergency messages for response prioritization</p>
            <a href="/message-classification" class="btn">Classify Messages</a>
        </div>
    </div>
    <script>
        const body = document.body;
        const toggleBtn = document.getElementById('theme-toggle');
        const currentTheme = localStorage.getItem('theme') || 'dark-theme';

        body.className = currentTheme;
        if (currentTheme === 'light-theme') {
            toggleBtn.textContent = '🌙 Dark Mode';
        } else {
            toggleBtn.textContent = '☀️ Light Mode';
        }

        toggleBtn.addEventListener('click', () => {
            if (body.classList.contains('dark-theme')) {
                body.classList.remove('dark-theme');
                body.classList.add('light-theme');
                toggleBtn.textContent = '🌙 Dark Mode';
                localStorage.setItem('theme', 'light-theme');
            } else {
                body.classList.remove('light-theme');
                body.classList.add('dark-theme');
                toggleBtn.textContent = '☀️ Light Mode';
                localStorage.setItem('theme', 'dark-theme');
            }
        });
    </script>
</body>
</html>"""
    return html_content

@app.route('/flood-prediction')
def flood_prediction():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Flood Risk Prediction</title>
    <style>
        /* Copy the full style block from the index route here */
        :root {
            --bg-start: #0f2027;
            --bg-end: #203a43;
            --text-color: #333;
            --container-bg: white;
            --btn-bg: #3498db;
            --btn-hover-bg: #2980b9;
            --result-bg: #ecf0f1;
            --link-color: #3498db;
        }
        
        body.dark-theme {
            --text-color: #f0f0f0;
            --container-bg: #2c3e50;
            --btn-bg: #1abc9c;
            --btn-hover-bg: #16a085;
            --result-bg: #34495e;
            --link-color: #1abc9c;
        }

        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: var(--bg-start);
            color: var(--text-color);
            transition: background-color 0.5s, color 0.5s;
        }
        .container { 
            max-width: 600px; 
            margin: 0 auto; 
            background: var(--container-bg); 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: background 0.5s;
        }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: var(--text-color); }
        input[type="range"] { width: 100%; }
        .range-value { float: right; font-weight: bold; color: #e74c3c; }
        .btn { 
            background: var(--btn-bg); 
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            width: 100%; 
            font-size: 16px; 
            transition: all 0.3s; 
        }
        .btn:hover { 
            background: var(--btn-hover-bg); 
            transform: translateY(-1px); 
        }
        .result { 
            margin-top: 20px; 
            padding: 20px; 
            background: var(--result-bg); 
            border-radius: 5px; 
            display: none;
            color: var(--text-color);
        }
        .risk-low { color: #27ae60; } .risk-medium { color: #f39c12; } .risk-high { color: #e74c3c; } .risk-very-high { color: #8e44ad; font-weight: bold; }
        a { color: var(--link-color); text-decoration: none; transition: color 0.5s;}
    </style>
    <script>
        // Function to set theme from local storage
        function applyTheme() {
            const currentTheme = localStorage.getItem('theme') || 'dark-theme';
            document.body.className = currentTheme;
        }
        // Apply theme on page load
        document.addEventListener('DOMContentLoaded', applyTheme);
    </script>
</head>
<body>
    <div class="container">
        <h1>🌊 Flood Risk Prediction</h1>
        <p>Adjust the environmental factors to predict flood risk probability</p>

        <form id="floodForm">
            <div class="form-group">
                <label>Monsoon Intensity (0.0 - 1.0): <span class="range-value" id="monsoon-val">0.5</span></label>
                <input type="range" name="MonsoonIntensity" min="0" max="1" step="0.01" value="0.5" oninput="document.getElementById('monsoon-val').textContent=this.value">
            </div>

            <div class="form-group">
                <label>Deforestation Level (0.0 - 1.0): <span class="range-value" id="deforest-val">0.5</span></label>
                <input type="range" name="Deforestation" min="0" max="1" step="0.01" value="0.5" oninput="document.getElementById('deforest-val').textContent=this.value">
            </div>

            <div class="form-group">
                <label>Urbanization (0.0 - 1.0): <span class="range-value" id="urban-val">0.5</span></label>
                <input type="range" name="Urbanization" min="0" max="1" step="0.01" value="0.5" oninput="document.getElementById('urban-val').textContent=this.value">
            </div>

            <div class="form-group">
                <label>River Management (0.0 - 1.0): <span class="range-value" id="river-val">0.5</span></label>
                <input type="range" name="RiverManagement" min="0" max="1" step="0.01" value="0.5" oninput="document.getElementById('river-val').textContent=this.value">
            </div>

            <div class="form-group">
                <label>Climate Change Impact (0.0 - 1.0): <span class="range-value" id="climate-val">0.5</span></label>
                <input type="range" name="ClimateChange" min="0" max="1" step="0.01" value="0.5" oninput="document.getElementById('climate-val').textContent=this.value">
            </div>

            <button type="submit" class="btn">🔍 Predict Flood Risk</button>
        </form>

        <div id="result" class="result">
            <h3 id="risk-level"></h3>
            <p id="confidence"></p>
            <div id="probabilities"></div>
        </div>

        <div style="text-align: center; margin-top: 20px;">
            <a href="/">← Back to Dashboard</a>
        </div>
    </div>

    <script>
    document.getElementById('floodForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());

        const defaults = {
            'TopographyDrainage': 0.5, 'DamsQuality': 0.5, 'Siltation': 0.5,
            'AgriculturalPractices': 0.5, 'Encroachments': 0.5, 'IneffectiveDisasterPreparedness': 0.5,
            'DrainageSystems': 0.5, 'CoastalVulnerability': 0.5, 'Landslides': 0.5,
            'Watersheds': 0.5, 'DeterioratingInfrastructure': 0.5, 'PopulationScore': 0.5,
            'WetlandLoss': 0.5, 'InadequatePlanning': 0.5, 'PoliticalFactors': 0.5
        };
        Object.assign(data, defaults);

        try {
            const response = await fetch('/api/predict-flood', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.error) {
                alert('Error: ' + result.error);
                return;
            }

            document.getElementById('risk-level').textContent = `Flood Risk: ${result.risk_level}`;
            document.getElementById('risk-level').className = `risk-${result.risk_level.toLowerCase().replace(' ', '-')}`;
            document.getElementById('confidence').textContent = `Confidence: ${result.confidence}%`;

            let probHtml = '<h4>Risk Level Breakdown:</h4>';
            for (const [level, prob] of Object.entries(result.probabilities)) {
                probHtml += `<p><strong>${level}:</strong> ${prob}%</p>`;
            }
            document.getElementById('probabilities').innerHTML = probHtml;

            document.getElementById('result').style.display = 'block';
            document.getElementById('result').scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            alert('Error: ' + error.message);
        }
    });
    </script>
</body>
</html>"""
    return html_content

@app.route('/message-classification')
def message_classification():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Message Classification</title>
    <style>
        /* Copy the full style block from the index route here */
        :root {
            --bg-start: #0f2027;
            --bg-end: #203a43;
            --text-color: #333;
            --container-bg: white;
            --btn-bg: #27ae60;
            --btn-hover-bg: #229954;
            --result-bg: #ecf0f1;
            --link-color: #27ae60;
            --category-bg: #3498db;
        }
        
        body.dark-theme {
            --text-color: #f0f0f0;
            --container-bg: #2c3e50;
            --btn-bg: #1abc9c;
            --btn-hover-bg: #16a085;
            --result-bg: #34495e;
            --link-color: #1abc9c;
            --category-bg: #3498db;
        }

        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: var(--bg-start);
            color: var(--text-color);
            transition: background-color 0.5s, color 0.5s;
        }
        .container { 
            max-width: 600px; 
            margin: 0 auto; 
            background: var(--container-bg); 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: background 0.5s;
        }
        textarea { 
            width: 100%; 
            height: 120px; 
            padding: 10px; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            font-size: 16px; 
            resize: vertical;
            background: var(--container-bg);
            color: var(--text-color);
            transition: background 0.5s, color 0.5s;
        }
        .btn { 
            background: var(--btn-bg); 
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            width: 100%; 
            font-size: 16px; 
            margin-top: 15px; 
            transition: all 0.3s;
        }
        .btn:hover { 
            background: var(--btn-hover-bg); 
            transform: translateY(-1px); 
        }
        .result { 
            margin-top: 20px; 
            padding: 20px; 
            background: var(--result-bg); 
            border-radius: 5px; 
            display: none;
            color: var(--text-color);
        }
        .category { 
            background: var(--category-bg); 
            color: white; 
            padding: 5px 10px; 
            margin: 5px; 
            border-radius: 15px; 
            display: inline-block; 
            font-size: 14px; 
        }
        .examples { 
            margin-top: 20px; 
            padding: 15px; 
            background: var(--result-bg); /* Use result bg for consistent theme */
            border-radius: 5px; 
            border-left: 4px solid var(--link-color); 
            transition: background 0.5s;
        }
        .example { 
            cursor: pointer; 
            padding: 8px; 
            margin: 5px 0; 
            border-radius: 3px; 
            background: var(--container-bg); /* Use container bg for cleaner contrast */
            border: 1px solid #ddd; 
            transition: all 0.3s;
            color: var(--text-color);
        }
        .example:hover { 
            background: #e9ecef; 
            transform: translateX(5px); 
        }
        a { color: var(--link-color); text-decoration: none; transition: color 0.5s;}
    </style>
    <script>
        // Function to set theme from local storage
        function applyTheme() {
            const currentTheme = localStorage.getItem('theme') || 'dark-theme';
            document.body.className = currentTheme;
        }
        // Apply theme on page load
        document.addEventListener('DOMContentLoaded', applyTheme);
    </script>
</head>
<body>
    <div class="container">
        <h1>💬 Disaster Message Classification</h1>
        <p>Enter an emergency message to automatically classify its type and priority level</p>

        <form id="messageForm">
            <textarea name="message" placeholder="Type your emergency message here..." required></textarea>
            <button type="submit" class="btn">🏷️ Classify Message</button>
        </form>

        <div id="result" class="result">
            <h3>📊 Classification Results:</h3>
            <p><strong>Original Message:</strong> <em id="original-message"></em></p>
            <p><strong>Categories Detected:</strong> <span id="category-count"></span></p>
            <div id="categories"></div>
        </div>

        <div class="examples">
            <h4>💡 Try these example messages:</h4>
            <div class="example" onclick="setMessage('We need water and food supplies urgently for flood victims')">🌊 Water and food needed for flood victims</div>
            <div class="example" onclick="setMessage('Medical help required for injured people in earthquake area')">🏥 Medical help needed for earthquake victims</div>
            <div class="example" onclick="setMessage('Shelter needed for homeless families after storm')">🏠 Shelter needed after storm damage</div>
            <div class="example" onclick="setMessage('Search and rescue teams needed in collapsed building')">🚁 Search and rescue needed</div>
            <div class="example" onclick="setMessage('Fire spreading rapidly, evacuation required')">🔥 Fire emergency evacuation</div>
        </div>

        <div style="text-align: center; margin-top: 20px;">
            <a href="/">← Back to Dashboard</a>
        </div>
    </div>

    <script>
    function setMessage(text) {
        document.querySelector('textarea[name="message"]').value = text;
    }

    document.getElementById('messageForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(e.target);
        const data = { message: formData.get('message') };

        try {
            const response = await fetch('/api/classify-message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.error) {
                alert('Error: ' + result.error);
                return;
            }

            document.getElementById('original-message').textContent = result.message;
            document.getElementById('category-count').textContent = result.total_categories;

            let categoriesHtml = '';
            if (result.predicted_categories.length > 0) {
                result.predicted_categories.forEach(category => {
                    categoriesHtml += `<span class="category">${category.replace('_', ' ')}</span>`;
                });
            } else {
                categoriesHtml = '<em style="color: #666;">No specific categories detected</em>';
            }

            document.getElementById('categories').innerHTML = categoriesHtml;
            document.getElementById('result').style.display = 'block';
            document.getElementById('result').scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            alert('Error: ' + error.message);
        }
    });
    </script>
</body>
</html>"""
    return html_content

# --- API ENDPOINTS ---

@app.route('/api/predict-flood', methods=['POST'])
def api_predict_flood():
    try:
        data = request.get_json()
        features = [
            float(data.get('MonsoonIntensity', 0.5)),
            float(data.get('TopographyDrainage', 0.5)),
            float(data.get('RiverManagement', 0.5)),
            float(data.get('Deforestation', 0.5)),
            float(data.get('Urbanization', 0.5)),
            float(data.get('ClimateChange', 0.5)),
            float(data.get('DamsQuality', 0.5)),
            float(data.get('Siltation', 0.5)),
            float(data.get('AgriculturalPractices', 0.5)),
            float(data.get('Encroachments', 0.5)),
            float(data.get('IneffectiveDisasterPreparedness', 0.5)),
            float(data.get('DrainageSystems', 0.5)),
            float(data.get('CoastalVulnerability', 0.5)),
            float(data.get('Landslides', 0.5)),
            float(data.get('Watersheds', 0.5)),
            float(data.get('DeterioratingInfrastructure', 0.5)),
            float(data.get('PopulationScore', 0.5)),
            float(data.get('WetlandLoss', 0.5)),
            float(data.get('InadequatePlanning', 0.5)),
            float(data.get('PoliticalFactors', 0.5))
        ]
        result = system.predict_flood_risk(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/classify-message', methods=['POST'])
def api_classify_message():
    try:
        data = request.get_json()
        message = data.get('message', '')
        if not message.strip():
            return jsonify({'error': 'Message cannot be empty'}), 400
        result = system.classify_message(message)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400