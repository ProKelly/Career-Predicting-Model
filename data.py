import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# reading the data from the csv file
df = pd.read_csv('career_datasets.csv')

# i'm encoding the features that are composed of strings
encoder = LabelEncoder()
for col in ['secondary_school_studies', 'interest', 'personality_trait', 'family_background', 'financial_status', 'work_environment', 'your_hobby']:
    df[col] = encoder.fit_transform(df[col])


# here i'm encoding the features from the data that are composed of lists 
mlb = MultiLabelBinarizer()
for col in ['favorite_subjects', 'subjects_passed']:
    df_encoded = pd.DataFrame(mlb.fit_transform(df[col]), columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop(col, axis=1), df_encoded], axis=1)

#here are splitting the data into features and the target
X = df.drop('career', axis=1)
y = df['career']

#here i'm splitting the data into training set and testing set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#here i'm training the machine learning model
model = LogisticRegression(max_iter=100000000, solver='sag')
model.fit(X_train, y_train)

#here i'm cumputing the accuracy of the model 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example of inputs for new data 
secondary_school_studies = 'Science'
favorite_subjects = 'Chemistry,Physics'.split(',')
subjects_passed = 'Mathematics,Physics'.split(',')
interest = 'Science'
personality_trait = 'Introverted'
family_background = 'Highly Educated'
financial_status = 'Financially Stable'
work_environment = 'Competitive'
your_hobby = 'Gardening'
expected_university_duration = 7
grades = 88

# Encode 'favorite_subjects' and 'subjects_passed'
favorite_subjects_encoded = mlb.transform([favorite_subjects])
subjects_passed_encoded = mlb.transform([subjects_passed])

# Print lengths of encoded favorite subjects and subjects passed
print("Length of favorite_subjects_encoded:", favorite_subjects_encoded.shape)
print("Length of subjects_passed_encoded:", subjects_passed_encoded.shape)

# Encode categorical features
secondary_school_studies_encoded = encoder.transform([secondary_school_studies])[0]
interest_encoded = encoder.transform([interest])[0]
personality_trait_encoded = encoder.transform([personality_trait])[0]
family_background_encoded = encoder.transform([family_background])[0]
financial_status_encoded = encoder.transform([financial_status])[0]
work_environment_encoded = encoder.transform([work_environment])[0]
your_hobby_encoded = encoder.transform([your_hobby])[0]

# Create DataFrame for new data
new_data = pd.DataFrame({
    'secondary_school_studies': [secondary_school_studies_encoded],
    'interest': [interest_encoded],
    'personality_trait': [personality_trait_encoded],
    'family_background': [family_background_encoded],
    'financial_status': [financial_status_encoded],
    'work_environment': [work_environment_encoded],
    'your_hobby': [your_hobby_encoded],
    'expected_university_duration': [expected_university_duration],
    'grades': [grades],
    **{interest: favorite_subjects_encoded[0, i] for i, interest in enumerate(mlb.classes_)},
    **{interest: subjects_passed_encoded[0, i] for i, interest in enumerate(mlb.classes_)}
})

# Ensure the same order of features as during training
new_data = new_data[X.columns]

# Make predictions
prediction = model.predict(new_data)
print("Predicted career:", prediction)








'''This was how i earlier collected the input from the user, 
but i latter just inserted the input directly as seen above'''

# Step 7: Make predictions for new data
# Collect input from user
# secondary_school_studies = input('Please enter your secondary school studies: ')
# favorite_subjects = input('Please enter your favorite subjects separated by comma (e.g., Chemistry,Mathematics): ').split(',')
# subjects_passed = input('Please enter subjects passed separated by comma (e.g., Physics,Chemistry): ').split(',')
# interest = input('Please enter your interest: ')
# personality_trait = input('Please enter your personality trait: ')
# family_background = input('Please enter your family background: ')
# financial_status = input('Please enter your financial status: ')
# work_environment = input('Please enter your work environment: ')
# your_hobby = input('Please enter your hobby: ')
# expected_university_duration = int(input('Please enter your expected university duration: '))
# grades = int(input('Please enter your grades: '))