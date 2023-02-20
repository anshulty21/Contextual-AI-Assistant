from flask import Flask,render_template,request
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
from string import punctuation


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/bert'
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), unique=True, nullable=False)

def inference(question, context):
  tokenizer = DistilBertTokenizer.from_pretrained("./models/tokenizer/")
  model = TFDistilBertForQuestionAnswering.from_pretrained("./models/") 
  input_dict = tokenizer.encode_plus(question, context, padding = 'max_length', max_length=128, return_tensors='tf')
  start_scores, end_scores  = model(input_dict)
  del model
  all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
  answer =  ' '.join(all_tokens[tf.math.argmax(start_scores, 1)[0] : tf.math.argmax(end_scores, 1)[0]+1])
  final_answer=""
  for char in answer:
      if char not in punctuation:
          final_answer=final_answer+char

  return final_answer

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form.get("username")
        email=request.form.get("email")
        password=request.form.get("password")
        user=User.query.filter_by(email=email).first()
        if user:
            print("user already Exist")
            return render_template("/signup.html",a="Email Already Exist")

        new_user=db.engine.execute(f"INSERT INTO `user`(`username`,`email`,`password`)VALUES('{username}','{email}','{password}')")
        return render_template("/login.html")   
    return render_template("signup.html")

@app.route("/login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        email=request.form.get("email")
        password=request.form.get("password")
        user=User.query.filter_by(email=email,password=password).first()
        if user:
            return render_template("/menu.html")
        else:
            return render_template("/login.html",a="Invald username & password")
    return render_template("login.html")



@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
      return render_template('predict.html')
    if request.method == 'POST':
      question, context = request.form['question'], request.form['context']
      final_answer = inference(question, context)
      return render_template('predict.html', result = final_answer, question=question, context=context)



if __name__ == "__main__":
    app.run(debug=True)