from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import json
import spellchecker 

pipeline = joblib.load('pipeline.sav')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


# @app.route('/api', methods=['POST'])
# def make_prediction():
#     data = request.get_json(force=True)
#     #convert our json to a numpy array
#     one_hot_data = input_to_one_hot(data)
#     predict_request = gbr.predict([one_hot_data])
#     output = [predict_request[0]]
#     print(data)
#     return jsonify(results=output)

@app.route('/api',methods=['POST'])
def get_delay():

	df = pd.read_csv('datatopredict.csv')
	result=request.form
	query = result['query']

	user_input = {'query':query}

	print(query)
	query = spellchecker.correctspell(query)
	#a = input_to_one_hot(user_input)
	pred = pipeline.predict([query])
	sub_df = df[df['class'] == pred[0]]

	resolution = sub_df['Resolution'].values[0]
	#print(type(sub_df['Resolution'][0]))
	
	#price_pred = round(price_pred, 2)
	#price_pred = 'A length of Cat5e cable, a crimping tool with cable stripper/cutter, 8P8C jacks (more than two - mistakes happen), plug 		boots, cable tester. All of these tools should be available at your local electronics parts and spare store, or Amazon.'
	return json.dumps({'price':resolution});
	# return render_template('result.html',prediction=price_pred)

if __name__ == '__main__':
    app.run(port=8080, debug=True)






