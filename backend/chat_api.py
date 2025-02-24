import re
from flask import Flask, request, Response
from flask_cors import CORS
from dotenv import load_dotenv
import os
import openai
import json
from consultent_agent import main

# Load environment variables
load_dotenv()

# Setup OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    print("Error: API key not found")
    exit(1)

openai.api_key = api_key

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Store sessions
sessions = {}

def save_to_file(session):
    """Save collected user information to a text file."""
    with open("user_input.txt", "a") as file:
        file.write(json.dumps(session, indent=4) + "\n")

def check_missing_info(session):
    """Check if all required information has been collected."""
    missing_info = []
    if 'size_of_restaurant' not in session or session['size_of_restaurant'] == '':
        missing_info.append("size of the restaurant (in square meters)")
    if 'total_area_eco_decorations' not in session or session['total_area_eco_decorations'] == '':
        missing_info.append("total area for eco-friendly decorations (in square meters)")
    if 'area_of_swimming_pool' not in session or session['area_of_swimming_pool'] == '':
        missing_info.append("area of the swimming pool (in square meters)")
    if 'area_of_restaurant' not in session or session['area_of_restaurant'] == '':
        missing_info.append("area of the restaurant (in square meters)")
    return missing_info

@app.route('/message', methods=['POST'])
def handle_message():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        user_message = data.get('message', '')
        
        if not user_message:
            return Response("Please provide a message", mimetype='text/plain')
        
        if user_id not in sessions:
            sessions[user_id] = {}
        
        session = sessions[user_id]
        
        # Prepare the OpenAI prompt
        messages = [
            {"role": "system", "content": "You are a business consultant helping clients gather necessary information. Your goal is to collect the following details: size of the restaurant (in square meters), total area for eco-friendly decorations (in square meters), area of the swimming pool (in square meters), and area of the restaurant (in square meters). Respond with the information in a valid JSON format, with keys 'size_of_restaurant', 'total_area_eco_decorations', 'area_of_swimming_pool', and 'area_of_restaurant'. If any information is missing, leave it blank, but ensure the response is still valid JSON."},
            {"role": "user", "content": user_message}
        ]
        
        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        bot_response = response.choices[0].message.content
        
        # Check if the response is empty
        if not bot_response or bot_response.strip() == "":
            return Response("I didn't understand that. Could you please try again?", mimetype='text/plain')
        
        # Clean the response
        cleaned_response = re.sub(r'```json\n|```', '', bot_response).strip()
        
        # Parse JSON response
        try:
            bot_data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            return Response("I didn't understand that. Could you please try again?", mimetype='text/plain')
        
        # Update session with new data
        if bot_data.get('size_of_restaurant'):
            session['size_of_restaurant'] = bot_data['size_of_restaurant']
        if bot_data.get('total_area_eco_decorations'):
            session['total_area_eco_decorations'] = bot_data['total_area_eco_decorations']
        if bot_data.get('area_of_swimming_pool'):
            session['area_of_swimming_pool'] = bot_data['area_of_swimming_pool']
        if bot_data.get('area_of_restaurant'):
            session['area_of_restaurant'] = bot_data['area_of_restaurant']
        
        # Check for missing information
        missing_info = check_missing_info(session)
        
        if not missing_info:
            # All information collected
            save_to_file(session)
            main()
            return Response(
                "Thank you! I have all the information I need. Is there anything else I can help you with?",
                mimetype='text/plain'
            )
        else:
            # Ask for missing information
            return Response(
                f"I still need the following details: {', '.join(missing_info)}. Please provide them one by one.",
                mimetype='text/plain'
            )

    except Exception as e:
        return Response(f"I encountered an error: {str(e)}. Could you please try again?", mimetype='text/plain')

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)