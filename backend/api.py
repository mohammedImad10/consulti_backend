import json
from flask import Flask, request, Response
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Store sessions
sessions = {}

# Questions to ask
questions = [
    "What is the size of the restaurant area in square meters (m2)?",
    "What is the total area in square meters (m) required for the eco-friendly decorations in the restaurant?",
    "What is the cost of eco-friendly decorations per m in NIS?",
    "What is the area of the swimming pool in square meters?",
    "What is the area of the restaurant in square meters (m2)?",
    "What is the total area of the restaurant in square meters?"
]

def save_to_file(session):
    """Save collected user information to a text file."""
    with open("user_input.txt", "a") as file:
        file.write(json.dumps(session, indent=4) + "\n")

@app.route('/message', methods=['POST'])
def handle_message():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        user_message = data.get('message', '')
        
        if not user_message:
            return Response("Please provide a message", mimetype='text/plain')
        
        if user_id not in sessions:
            sessions[user_id] = {'question_index': 0, 'answers': {}}
        
        session = sessions[user_id]
        
        # Get the current question to ask
        question_index = session['question_index']
        
        if question_index >= len(questions):
            # All questions have been answered
            save_to_file(session)
            return Response("Great! I got your information. and now i will process the files", mimetype='text/plain')
        
        # Ask 3 questions at a time
        questions_to_ask = questions[question_index:question_index+3]
        question_text = "\n".join(questions_to_ask)
        
        # Save the user's answers
        session['answers'][f'question_{question_index+1}'] = user_message
        
        # Move to the next set of questions
        session['question_index'] += 3
        
        return Response(f"Thank you for your answers! Here are the next questions:\n{question_text}", mimetype='text/plain')

    except Exception as e:
        return Response("I encountered an error. Could you please try again?", mimetype='text/plain')

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
