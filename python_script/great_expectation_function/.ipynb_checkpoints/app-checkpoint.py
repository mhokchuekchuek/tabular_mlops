import os
from flask import Flask,render_template

import json
with open('path.json') as json_file:
    data = json.load(json_file)

app = Flask('app',template_folder=data["flask_path"])

@app.route('/')
def main():
    return render_template(data["html_path"])
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
        