from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model once when the app starts
loaded_model = pickle.load(open("pathwise.pkl", 'rb'))

@app.route('/')
def career():
    return render_template("hometest.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        print("Raw input:", result)

        try:
            # Convert form data to a numeric array
            arr = [float(value) for value in result.values() if value]  # Ensure conversion to float and skip empty values
            if not arr:
                return "No valid input provided."

            data = np.array(arr).reshape(1, -1)  # Reshape for model input
            print("Input data:", data)

            # Make predictions
            predictions = loaded_model.predict(data)
            pred_proba = loaded_model.predict_proba(data)

            # Filter predictions
            pred = pred_proba > 0.05
            final_res = {}

            # Create result mapping
            jobs_dict = {
                0: 'AI ML Specialist',
                1: 'API Integration Specialist',
                2: 'Application Support Engineer',
                3: 'Business Analyst',
                4: 'Customer Service Executive',
                5: 'Cyber Security Specialist',
                6: 'Data Scientist',
                7: 'Database Administrator',
                8: 'Graphics Designer',
                9: 'Hardware Engineer',
                10: 'Helpdesk Engineer',
                11: 'Information Security Specialist',
                12: 'Networking Engineer',
                13: 'Project Manager',
                14: 'Software Developer',
                15: 'Software Tester',
                16: 'Technical Writer'
            }

            # Fill final results
            index = 0
            for j in range(17):
                if pred[0, j] and j != predictions[0]:  # Exclude the predicted job
                    final_res[index] = j
                    index += 1

            # Return results to the template
            data1 = predictions[0]
            return render_template("testafter.html", final_res=final_res, job_dict=jobs_dict, job0=data1)

        except ValueError as e:
            print("Error in data conversion:", e)
            return "Invalid input data, please enter numeric values."
        except Exception as e:
            print("Unexpected error:", e)
            return "An error occurred during prediction."

if __name__ == '__main__':
    app.run(debug=True)


