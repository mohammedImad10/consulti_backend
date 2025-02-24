import os
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import asyncio
from consultent_agent_temp import main, set_user_response
from flask_cors import CORS
import json


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")#async_mode='threading')
CORS(app)
loop = asyncio.new_event_loop()
in_process = False
input_info = ""

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    user_message = data.get('message')
    print('User message:', user_message)
    
    if user_message.lower() == 'finished':
        global in_process
        in_process = True
        socketio.start_background_task(run_main, socketio)
        return jsonify({'response': 'Starting the process...'})
    
    return jsonify({'response': "more information about bisuness or finished to start the process"})

def run_main(socketio):
    with app.app_context():  # Ensure Flask context
        
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main(socketio))

def save_to_file():
    """Save collected user information to a text file."""
    with open("customer_data_docs/user_input_2.txt", "a") as file:
        file.write( input_info + "\n")


@socketio.on('user_response')
def handle_user_response(data):
    print('User response:', data)
    response = data.get('response')
    user_message = data.get('response')
    
    if user_message.lower() == 'finished':
        save_to_file()
        global in_process
        in_process = True
        emit('message', {'info': 'Starting the process...'})
        socketio.start_background_task(run_main, socketio)
    else:
        if in_process:
            set_user_response(response= response, loop=loop)
        else:
            global input_info
            input_info +=  response + "\n"
            emit('message', {'question': 'give me more information about bisuness or type finished to start the process'})
        

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    socketio.run(app,host="0.0.0.0", port=port, debug=False)
    