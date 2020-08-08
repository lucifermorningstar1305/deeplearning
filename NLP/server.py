import os
from save_text import WikipediaSave
from flask import Flask, request
import json

app = Flask(__name__)

@app.route("/",methods=["GET"])
def hello_world():
    return json.dumps({"message" : "Hello World", "status":200})

@app.route("/search", methods=["POST"])
def search():
    return WikipediaSave().saveData(request.form["entity"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
