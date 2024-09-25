#!/usr/bin/python3

#
#
# if __name__ == "__main__":
#


import pymongo
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io

app = Flask(__name__)


client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["hackbattle"]

# employer   = db["employer"]
# applicants = db["applicants"]
# matches    = db["matches"]


EMPLOYER_ID  = 1
APPLICANT_ID = 0

# Landing page
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle GET and POST requests
@app.route('/employer', methods=['GET', 'POST'])
def employer_login():
    global EMPLOYER_ID

    if request.method == 'POST':
        name = request.form['company_name']
        location = request.form['location']
        description = request.form['description']
        social_media = request.form['social_media']

        logo_bytes = io.BytesIO()
        logo = request.files.get('logo', '')
        logo.save(logo_bytes)

        db.employer.insert_one(
	    { "_id": EMPLOYER_ID
	    , "name": name
	    , "location": location
	    , "description": description
	    , "social_media": social_media
	    , "logo": logo_bytes.getvalue()
	    , "positions": []
	    }
	)

        EMPLOYER_ID += 1

        return f'Hello, {name}!'
    else:
        return render_template('employer_login.html')

# Route to handle GET and POST requests
@app.route('/applicant', methods=['GET', 'POST'])
def applicant():

    global APPLICANT_ID

    if request.method == 'POST':


        print(request.form)
        name = request.form['name']
        email = request.form['email']

        logo_bytes = io.BytesIO()
        logo = request.files.get('logo-upload', '')
        logo.save(logo_bytes)

        pdf_bytes = io.BytesIO()
        pdf = request.files.get('pdf-upload', '')
        pdf.save(pdf_bytes)

        db.applicants.insert_one(
	    { "_id": APPLICANT_ID
	    , "name": name
	    , "email": email
	    , "logo": logo_bytes.getvalue()
	    , "resume": pdf_bytes.getvalue()
	    }
	)

        APPLICANT_ID += 1

        return f'Hello applicant, {name}!'
    else:
        return render_template('applicant_login.html')

# Dynamic URL with parameters
@app.route('/employer/<name>')
def show_employer_profile(name):
    index = db.employer.find_one({ "name": name })
    return render_template('employer_profile.html', company_name=index["name"], location=index["location"])

# Dynamic URL with parameters
@app.route('/user/<username>')
def show_user_profile(username):
    return f'User: {username}'

# Dynamic URL with parameters
@app.route('/employer/<employer>/<post_id>')
def show_post_id(employer,post_id):
    return f'Employer: {employer}, post_id: {post_id}'

if __name__ == '__main__':
    app.run(debug=True)
