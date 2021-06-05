from flask import Flask,request,jsonify,render_template
from flask_cors import CORS, cross_origin
from TestingPipeline import getPrediction
from Declaration.Declaration import perform_test,BERTClass,read_pretrained_model_tokenizer
import Declaration.Declaration
from Declaration import Config

# import nltk
# nltk.download('punkt')


app = Flask(__name__)
CORS(app)

# model = Ner("out_base")

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    text = request.json["text"]
    try:
        # out = model.predict(text)
        out = getPrediction(text)
        return jsonify({"result":out})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    # app.run('127.0.0.0',port=5000)
    app.run('localhost', port=7000)

