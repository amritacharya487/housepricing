import pickle 
from flask import Flask , app , request , url_for , render_template 
import numpy as np 
import pandas as pd 



app = Flask(__name__)
# load the model 
regmodel =pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home() :
    return render_template(home.html)

@app.route('/predct_api' , Method =[Post]) 
def predict_api():
    data = request.jason['data']
    print(data)
    print(np.array(list(data.value())).reshape(1 ,-1))
    new_data = scale.transform(np.array(list(data.value())).reshape(1 ,-1))
    output = regmodel.score(new_data)
    print(output[0])
    return jsoinfy(output[0])


if __name__ =="__main__":
   app.run(Debug=True)
