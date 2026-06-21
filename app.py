# ───────────────────────────────────────────────────────────────────────────────────
# Now we'll import all necessary libraries including gradio and load the saved model.
# ───────────────────────────────────────────────────────────────────────────────────
import gradio as gradio_library
import pickle
import pandas as pd


try:
    with open('best_loan_model.pkl', 'rb') as model_file:
        gradio_loaded_model = pickle.load(model_file)
    print("Model loaded successfully.")

except FileNotFoundError:
    print("ERROR: 'best_loan_model.pkl' not found. Please check the file path.")
    gradio_loaded_model = None

except Exception as unexpected_error:
    print(f"ERROR: Unexpected problem while loading model: {unexpected_error}")
    gradio_loaded_model = None





# ─────────────────────────────────────────────────────────────────
# Now we'll define the prediction function for Gradio.
# It takes all user inputs and returns a prediction string.
# ─────────────────────────────────────────────────────────────────
def predict_loan_approval(gender, married, dependents, education,
                           self_employed, loan_amount, loan_amount_term,
                           credit_history, property_area,
                           applicant_income, coapplicant_income):

    # ─────────────────────────────────────────────────────────────────
    # Now we'll check if the model is loaded before doing anything.
    # ─────────────────────────────────────────────────────────────────
    if gradio_loaded_model is None:
        return "ERROR: Model is not loaded. Please check the server logs."

    try:
        # ───────────────────────────────────────────────────────────────────
        # Now we'll combine applicant and co-applicant income into one value.
        # ───────────────────────────────────────────────────────────────────
        total_income_value = applicant_income + coapplicant_income

        # ─────────────────────────────────────────────────────────────────
        # Now we'll build a single-row DataFrame from the user inputs.
        # Column names must match exactly what the model was trained on.
        # ─────────────────────────────────────────────────────────────────
        input_dataframe = pd.DataFrame([{
            'Gender'          : gender,
            'Married'         : married,
            'Dependents'      : dependents,
            'Education'       : education,
            'Self_Employed'   : self_employed,
            'LoanAmount'      : loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History'  : credit_history,
            'Property_Area'   : property_area,
            'Total_Income'    : total_income_value
        }])

        # ─────────────────────────────────────────────────────────────────
        # Now we'll get the predicted class (0 or 1) and probabilities.
        # ─────────────────────────────────────────────────────────────────
        predicted_class       = gradio_loaded_model.predict(input_dataframe)[0]
        predicted_probability = gradio_loaded_model.predict_proba(input_dataframe)[0]

        # ─────────────────────────────────────────────────────────────────
        # Now we'll build a human-readable result string to return.
        # ─────────────────────────────────────────────────────────────────
        if predicted_class == 1:
            result_text = f"✅ LOAN APPROVED\nConfidence: {predicted_probability[1]*100:.1f}%"
        else:
            result_text = f"❌ LOAN REJECTED\nConfidence: {predicted_probability[0]*100:.1f}%"

        return result_text

    except ValueError as value_error:
        return f"ERROR: Invalid input value — {value_error}"

    except Exception as unexpected_error:
        return f"ERROR: Something went wrong during prediction — {unexpected_error}"


print("Prediction function defined.")





# ──────────────────────────────────────────────────────────────────────────────────────
# Now we'll create the try-catch block for the Gradio interface with all input fields.
# ──────────────────────────────────────────────────────────────────────────────────────
try:

    # ─────────────────────────────────────────────────────────────────
    # Now we'll build the Gradio interface with all input fields.
    # ─────────────────────────────────────────────────────────────────
    gradio_interface = gradio_library.Interface(
        fn=predict_loan_approval,
        inputs=[
            gradio_library.Dropdown(
                choices=['Male', 'Female'],
                label='Gender',
                value='Male'
            ),
            gradio_library.Dropdown(
                choices=['Yes', 'No'],
                label='Married',
                value='Yes'
            ),
            gradio_library.Dropdown(
                choices=['0', '1', '2', '3+'],
                label='Dependents',
                value='0'
            ),
            gradio_library.Dropdown(
                choices=['Graduate', 'Not Graduate'],
                label='Education',
                value='Graduate'
            ),
            gradio_library.Dropdown(
                choices=['Yes', 'No'],
                label='Self Employed',
                value='No'
            ),
            gradio_library.Number(
                label='Loan Amount (thousands)',
                value=150,
                minimum=1,
                maximum=700
            ),
            gradio_library.Number(
                label='Loan Amount Term (months)',
                value=360,
                minimum=12,
                maximum=480
            ),
            gradio_library.Dropdown(
                choices=[1.0, 0.0],
                label='Credit History (1 = Good, 0 = Bad)',
                value=1.0
            ),
            gradio_library.Dropdown(
                choices=['Urban', 'Semiurban', 'Rural'],
                label='Property Area',
                value='Urban'
            ),
            gradio_library.Number(
                label='Applicant Monthly Income',
                value=5000,
                minimum=0
            ),
            gradio_library.Number(
                label='Co-Applicant Monthly Income',
                value=0,
                minimum=0
            )
        ],
        outputs=gradio_library.Textbox(
            label='Loan Prediction Result',
            lines=2
        ),
        title='Loan Approval Prediction System',
        description=(
            'Enter the loan applicant details below to predict '
            'whether the loan will be approved or rejected.'
        ),
        examples=[
            ['Male', 'Yes', '0', 'Graduate', 'No', 150, 360, 1.0, 'Urban', 5000, 1500],
            ['Female', 'No', '2', 'Not Graduate', 'Yes', 300, 360, 0.0, 'Rural', 2000, 0]
        ]
    )

    print("Gradio interface created.")
    print("Set LAUNCH_GRADIO = True to launch the web app.")

    # ─────────────────────────────────────────────────────────────────
    # Now we'll launch the Gradio app.
    # LAUNCH_GRADIO = False means it won't auto-launch.
    # Set it to True when you want to open the app.
    # share=True creates a public link (useful for Colab).
    # ─────────────────────────────────────────────────────────────────
    LAUNCH_GRADIO = True

    if LAUNCH_GRADIO:
        gradio_interface.launch(share=True)
    else:
        print("Gradio app is ready but NOT launched.")
        print("Set LAUNCH_GRADIO = True to start the web interface.")


except ImportError:
    print("Gradio is not installed. Install it with:  pip install gradio")




