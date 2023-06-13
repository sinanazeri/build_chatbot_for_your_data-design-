import base64
import json
import logging
import os

from flask import Flask, render_template, request
from flask_cors import CORS

import worker

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process-message', methods=['POST'])
def process_prompt_route():
    user_message = request.json['userMessage']
    print('user_message', user_message)

    bot_response = worker.process_prompt(user_message)
    bot_response = os.linesep.join([s for s in bot_response.splitlines() if s])

    response = app.response_class(
        response=json.dumps({"botResponse": bot_response}),
        status=200,
        mimetype='application/json'
    )

    print(response)
    return response


@app.route('/process-document', methods=['POST'])
def process_document_data():
    data = request.get_json()

    if 'fileData' not in data:
        return app.response_class(
            response=json.dumps({
                "botResponse": "It seems like the file was not uploaded correctly, can you try "
                               "again. If the problem persists, try using a different file"}),
            status=400,
            mimetype='application/json'
        )

    file_data = data['fileData']

    # If you have a Base64 string, the file data will have a prefix like 'data:application/pdf;base64,'.
    # We need to remove this prefix to get the actual Base64 string
    base64_string = file_data.split(',', 1)[-1]

    # Now decode the Base64 string back into bytes
    document = base64.b64decode(base64_string)

    worker.process_document(document)

    response = app.response_class(
        response=json.dumps({
            "botResponse": "Thank you for providing you pdf document. I have it analyzed it, so now you can ask me any "
                           "questions regarding it!"}),
        status=200,
        mimetype='application/json'
    )

    return response


if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')
