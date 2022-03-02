from flask import *
from predict_result import predict_result

app=Flask(__name__)

"""
@app.route("/")
def hello():
    return "Hello World"
"""
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method=="GET":
        return render_template("predict.html")
    elif request.method=="POST":
        url=request.form["url"]
        
        return render_template("predict.html",prediction=predict_result(url))

if __name__=="__main__":
    app.run()
