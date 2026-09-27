import pandas as pd
from flask import Flask, request, render_template
import pickle
app = Flask(__name__,template_folder="templates")
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl','rb'))
@app.route('/')
def h():
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/index')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    name = form_data.get('name', '').strip()
    skill_areas = [skill.strip() for skill in form_data.get('skills', '').split(',') if skill.strip()]

    try:
        cgpa = float(form_data.get('cgpa', ''))
        projects = int(form_data.get('projects', ''))
        workshops = int(form_data.get('workshops', ''))
        mini_projects = int(form_data.get('mini_projects', ''))
        communication_skills = float(form_data.get('communication_skills', ''))
        internship = int(form_data.get('internship', ''))
        hackathon = int(form_data.get('hackathon', ''))
        tw_percentage = float(form_data.get('tw_percentage', ''))
        te_percentage = float(form_data.get('te_percentage', ''))
        backlogs = int(form_data.get('backlogs', ''))
    except ValueError:
        return render_template('index.html', error='Enter a valid value for every field.', form_data=form_data), 400

    if not name or not skill_areas:
        return render_template('index.html', error='Add your name and at least one skill area.', form_data=form_data), 400
    if not 0 <= cgpa <= 10 or not 0 <= communication_skills <= 10:
        return render_template('index.html', error='CGPA and communication rating must be between 0 and 10.', form_data=form_data), 400
    if not 0 <= tw_percentage <= 100 or not 0 <= te_percentage <= 100:
        return render_template('index.html', error='School percentages must be between 0 and 100.', form_data=form_data), 400
    if min(projects, workshops, mini_projects, backlogs) < 0 or internship not in (0, 1) or hackathon not in (0, 1):
        return render_template('index.html', error='Counts cannot be negative; choose Yes or No for experience fields.', form_data=form_data), 400

    feature_names = [
        'CGPA', 'Major Projects', 'Workshops/Certificatios', 'Mini Projects',
        'Skills', 'Communication Skill Rating', 'Internship', 'Hackathon',
        '12th Percentage', '10th Percentage', 'backlogs',
    ]
    features = pd.DataFrame([[
        cgpa, projects, workshops, mini_projects, len(skill_areas),
        communication_skills, internship, hackathon, tw_percentage,
        te_percentage, backlogs,
    ]], columns=feature_names)
    placement_status = str(model.predict(features)[0])
    is_placed = placement_status.strip().lower() == 'placed'
    class_names = [str(label).strip().lower() for label in model.classes_]
    placement_score = float(model.predict_proba(features)[0][class_names.index('placed')]) * 100

    salary_estimate = None
    if is_placed:
        salary_features = features.assign(PlacementStatus=1.0)
        salary_estimate = float(model1.predict(salary_features)[0])

    summary = [
        {'label': 'CGPA', 'value': f'{cgpa:g} / 10'},
        {'label': 'Skill areas', 'value': str(len(skill_areas))},
        {'label': 'Major projects', 'value': str(projects)},
        {'label': 'Certifications', 'value': str(workshops)},
        {'label': 'Internship', 'value': 'Yes' if internship else 'No'},
        {'label': 'Hackathon', 'value': 'Yes' if hackathon else 'No'},
        {'label': '12th percentage', 'value': f'{tw_percentage:g}%'},
        {'label': '10th percentage', 'value': f'{te_percentage:g}%'},
        {'label': 'Backlogs', 'value': str(backlogs)},
    ]
    return render_template(
        'output.html',
        name=name,
        is_placed=is_placed,
        placement_status='Placed' if is_placed else 'Not placed',
        placement_score=placement_score,
        salary_estimate=salary_estimate,
        salary_label=f'INR {salary_estimate / 100000:g} lakh / year' if salary_estimate else None,
        summary=summary,
    )

if __name__ == "__main__":
    app.run(debug=True)