from flask import Flask, Response
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)

@app.route('/run-agent', methods=['POST'])
def run_agent():
    try:
        # Run the main function of consultant_agent.py asynchronously
        subprocess.Popen(["python", "consultant_agent.py"])
        return Response("Agent started successfully", mimetype='text/plain')
    except Exception as e:
        return Response(f"Error: {str(e)}", mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
