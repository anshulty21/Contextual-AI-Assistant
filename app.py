import re
from flask import Flask, render_template, request, send_from_directory, redirect, session, flash, url_for , jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, UserMixin, login_required, current_user
from transformers import  AutoTokenizer, AutoModelForQuestionAnswering
from string import punctuation
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import torch


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.static_folder = 'static'
app.static_url_path = '/static'

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/bert'
db = SQLAlchemy(app)

# set up the login manager
login_manager = LoginManager()
login_manager.init_app(app)

# define the user model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized():
    return redirect('/login')


tokenizer = AutoTokenizer.from_pretrained("Aaroosh/bert-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("Aaroosh/bert-finetuned-squad")

@app.route('/inference', methods=['POST'])
def inference():
    data = request.get_json()
    if not data or 'question' not in data or 'context' not in data:
        return jsonify({'error': 'Invalid request data'})
    question = data['question']
    context = data['context']
    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    result = {'answer': answer}
    return jsonify(result)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/signup", methods=['POST','GET'])
def signup():
    if current_user.is_authenticated:
        return redirect('/predict')

    if request.method == 'POST':
        email = request.form.get("email")
        password = request.form.get("password")

        # Validate the email address using regex
        pattern = r"[^@]+@[^@]+\.[^@]+"
        if not re.match(pattern, email):
            return render_template("signup.html", a="Invalid email address")

        user = User.query.filter_by(email=email).first()
        if user:
            return render_template("signup.html", a="Email Already Exists")

        hashed_password = generate_password_hash(password)
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        # Log in the new user
        login_user(new_user)

        # Redirect to the login page
        return redirect('/login')

    return render_template("signup.html")
    

@app.route("/login", methods=['POST', 'GET'])
def login():
    if current_user.is_authenticated:
        return redirect('/predict')

    if request.method == 'POST':
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            session['user_id'] = user.id
            return redirect('/predict')
        else:
            return render_template("login.html", a="Invalid email or password")

    return render_template("login.html")

@app.before_request
def check_login():
    if current_user.is_authenticated and request.path == '/':
        return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        # Get the username part of the user's email address
        email_parts = current_user.email.split('@')
        email_username = email_parts[0]
        
        return render_template('predict.html', email_username=email_username)
    
    if request.method == 'POST':
        # Check if question and context are present in the form data
        if 'question' not in request.form or 'context' not in request.form:
            return render_template('predict.html', error='Invalid input')

        # Get the question and context from the form data
        question = request.form['question']
        context = request.form['context']
        
        # Check if the question or context are empty
        if not question.strip() or not context.strip():
            return render_template('predict.html', error='Question and context cannot be empty')

        # Tokenize the input and generate the answer
        inputs = tokenizer(question, context, return_tensors='pt')
        outputs = model(**inputs)
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        
        # Render the result
        return render_template('predict.html', q=question, c=context, a=answer)
        
@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', current_user=current_user)


@app.route('/update_password', methods=['POST'])
@login_required
def update_password():
    # Get the entered passwords from the form
    current_password = request.form['current_password']
    new_password = request.form['new_password']

    # Retrieve the current user's record from the database
    user = User.query.get(current_user.id)

    # Check that the current password is valid
    if not user.verify_password(current_password):
        flash('Incorrect current password.')
        return redirect(url_for('profile'))

    # Update the user's password in the database
    user.set_password(new_password)
    db.session.commit()

    flash('Password updated successfully.')
    return redirect(url_for('profile'))

@app.route('/billing')
@login_required
def billing():
    return render_template('billing.html', current_user=current_user)

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    session.pop('user_id', None)
    return redirect('/login')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)