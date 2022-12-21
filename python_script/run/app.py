import os
import sys
sys.path.append('../..')
sys.path.append('/code/notebook/')
from flask import Flask,render_template

import json
with open('/code/notebook/script/great_expectation_function/path.json') as json_file:
    data = json.load(json_file)

with open('/code/notebook/script/eda/path.json') as json_file_1:
    data_1 = json.load(json_file_1)
    
app = Flask('app',template_folder=data["flask_path"])
app1 = Flask('app',template_folder=data_1["flask_path"])
@app.route('/')
def main():
    return render_template(data["html_path"])
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    app1.run(host='0.0.0.0', port=8081)