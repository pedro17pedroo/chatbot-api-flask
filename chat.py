import random
import json
import codecs

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask("chat")
CORS(app)


     
@app.route("/get", methods=["GET", "POST"])
def postRequestsms():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r', encoding='utf-8') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']

    model_state = data["model_state"]
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    bot_name = "Publinild"
    while True:
        # sentence = "do you use credit cards?"
        # userText = request.get_json()
        userText = request.args.get('msg')
        sentence = userText
        
    
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        # if prob.item() > 0.80:
        #    for intent in intents['intents']:
        #        if tag == intent["tag"]:
        #            return jsonify({"bootresposta":random.choice(intent['responses'])})
        # else:
        #    return jsonify({"bootresposta": "_false_"})
           
        if prob.item() > 0.80:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                   
                    return random.choice(intent['responses'])
        else:
            return  "_false_"
           
          
          
@app.route("/satisfiedQuestion", methods=["GET", "POST"])
def satisfiedQuestionSave():
    userText = request.args.get('msg')
    arquivo = open("satisfiedQuestion.txt", "a")
    arquivo.write(userText+" \n\n")
    return  "ok"

@app.route("/unsatisfiedQuestion", methods=["GET", "POST"])
def unsatisfiedQuestionSave():
    userText = request.args.get('msg')
    arquivo = open("unsatisfiedQuestion.txt", "a")
    arquivo.write(userText+" \n\n")
    return  "ok"
              
          
          
app.run(host='0.0.0.0', port='5000', debug=True)