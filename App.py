import openai
import os
import pandas as pd
from dotenv import load_dotenv
import gradio as gr
import re

# Load environment variables from .env
load_dotenv()

# Configure OpenAI API key
openai.api_key = ""

CSV_FILE = 'appointments.csv'
COLUMNS = ['Doctor', 'Date', 'Start Time', 'End Time', 'Patient']

def read_appointments():
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=COLUMNS)
    return df

def save_appointments(df):
    df.to_csv(CSV_FILE, index=False)

def add_appointment(doctor, date, start_time, end_time, patient):
    df = read_appointments()

    try:
        new_start = pd.to_datetime(f"{date} {start_time}", format="%Y-%m-%d %I:%M %p")
        new_end = pd.to_datetime(f"{date} {end_time}", format="%Y-%m-%d %I:%M %p")
    except ValueError:
        return False, "Invalid date or time format. Please use YYYY-MM-DD for dates and HH:MM AM/PM for times."

    df['Start DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Start Time'], format="%Y-%m-%d %I:%M %p", errors='coerce')
    df['End DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['End Time'], format="%Y-%m-%d %I:%M %p", errors='coerce')

    conflict = df[
        (df['Doctor'].str.lower() == doctor.lower()) &
        (df['Date'] == date) &
        ((new_start < df['End DateTime']) & (new_end > df['Start DateTime']))
    ]

    if not conflict.empty:
        return False, "There is a conflict with an existing appointment."

    df = df.drop(['Start DateTime', 'End DateTime'], axis=1)

    new_appointment = {
        'Doctor': doctor.title(),
        'Date': date,
        'Start Time': start_time,
        'End Time': end_time,
        'Patient': patient.title()
    }
    df = pd.concat([df, pd.DataFrame([new_appointment])], ignore_index=True)
    save_appointments(df)
    return True, "Appointment added successfully."

def edit_appointment(patient, new_date=None, new_start_time=None, new_end_time=None):
    df = read_appointments()
    appointment = df[df['Patient'].str.lower() == patient.lower()]
    
    if appointment.empty:
        return False, "No appointment found for this patient."
    
    idx = appointment.index[0]
    
    if new_date:
        df.at[idx, 'Date'] = new_date
    if new_start_time:
        df.at[idx, 'Start Time'] = new_start_time
    if new_end_time:
        df.at[idx, 'End Time'] = new_end_time
    
    if new_date or new_start_time or new_end_time:
        try:
            updated_start = pd.to_datetime(f"{df.at[idx, 'Date']} {df.at[idx, 'Start Time']}", format="%Y-%m-%d %I:%M %p")
            updated_end = pd.to_datetime(f"{df.at[idx, 'Date']} {df.at[idx, 'End Time']}", format="%Y-%m-%d %I:%M %p")
        except ValueError:
            return False, "Invalid date or time format. Please use YYYY-MM-DD for dates and HH:MM AM/PM for times."

        df['Start DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Start Time'], format="%Y-%m-%d %I:%M %p", errors='coerce')
        df['End DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['End Time'], format="%Y-%m-%d %I:%M %p", errors='coerce')

        conflict = df[
            (df['Doctor'].str.lower() == df.at[idx, 'Doctor'].lower()) &
            (df['Date'] == df.at[idx, 'Date']) &
            (df.index != idx) &
            ((updated_start < df['End DateTime']) & (updated_end > df['Start DateTime']))
        ]

        if not conflict.empty:
            return False, "There is a conflict with an existing appointment after the update."

        df = df.drop(['Start DateTime', 'End DateTime'], axis=1)

    save_appointments(df)
    return True, "Appointment updated successfully."

def update_appointment(patient, updated_details):
    df = read_appointments()
    appointment = df[df['Patient'].str.lower() == patient.lower()]
    if appointment.empty:
        return False, "No appointment found for this patient."

    idx = appointment.index[0]

    for field, value in updated_details.items():
        if field in COLUMNS and value:
            df.at[idx, field] = value

    save_appointments(df)
    return True, "Appointment updated successfully."

def delete_appointment(patient):
    df = read_appointments()
    df = df[df['Patient'].str.lower() != patient.lower()]
    save_appointments(df)
    return True, "Appointment deleted successfully."

def list_appointments():
    df = read_appointments()
    if df.empty:
        return "No appointments scheduled."
    return df.to_string(index=False)

def extract_appointment_details(user_input):
    prompt = f"""
Extract the following details from the user input:
- Action: (add, edit, list, update, delete)
- Doctor Name: 
- Date (YYYY-MM-DD): 
- Start Time (HH:MM AM/PM): 
- End Time (HH:MM AM/PM): 
- Patient Name: 

User Input: "{user_input}"

Response Format:
Action: <action>
Doctor Name: <doctor>
Date: <date>
Start Time: <start_time>
End Time: <end_time>
Patient Name: <patient>
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for scheduling doctor appointments."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            temperature=0.3,
        )
        text = response.choices[0].message['content'].strip()
        details = {}
        for line in text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                details[key.strip().lower().replace(' ', '_')] = value.strip()
        return details
    except Exception as e:
        return {"error": str(e)}

def chatbot_interface(user_input, state):
    # Ensure state is a dictionary
    if not isinstance(state, dict):
        state = {}

    # Initialize or retrieve histories
    gradio_chat_history = state.get("gradio", [])
    openai_chat_history = state.get("openai", [])

    # Append user input to OpenAI chat history
    openai_chat_history.append({"role": "user", "content": user_input})

    # Process user input and generate response
    details = extract_appointment_details(user_input)
    if "error" in details:
        assistant_response = f"Error extracting details: {details['error']}"
    else:
        action = details.get('action', '').lower()
        doctor = details.get('doctor_name', '')
        date = details.get('date', '')
        start_time = details.get('start_time', '')
        end_time = details.get('end_time', '')
        patient = details.get('patient_name', '')

        if action == 'add':
            if not all([doctor, date, start_time, end_time, patient]):
                assistant_response = "Please provide all the necessary details to add an appointment."
            else:
                success, message = add_appointment(doctor, date, start_time, end_time, patient)
                assistant_response = message
        elif action == 'edit':
            if not patient:
                assistant_response = "Please provide the patient's name to edit their appointment."
            else:
                success, message = edit_appointment(patient, new_date=date, new_start_time=start_time, new_end_time=end_time)
                assistant_response = message
        elif action == 'update':
            if not patient:
                assistant_response = "Please provide the patient's name to update their appointment."
            else:
                updated_details = {
                    'Doctor': doctor,
                    'Date': date,
                    'Start Time': start_time,
                    'End Time': end_time,
                }
                success, message = update_appointment(patient, updated_details)
                assistant_response = message
        elif action == 'delete':
            if not patient:
                assistant_response = "Please provide the patient's name to delete their appointment."
            else:
                success, message = delete_appointment(patient)
                assistant_response = message
        elif action == 'list':
            assistant_response = list_appointments()
        else:
            assistant_response = "I'm sorry, I couldn't understand your request. Please try again."

    # Append interaction to Gradio chat history
    gradio_chat_history.append({"role": "user", "content": user_input})
    gradio_chat_history.append({"role": "assistant", "content": assistant_response})

    # Update and return the state
    state = {"gradio": gradio_chat_history, "openai": openai_chat_history}
    return gradio_chat_history, state



gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.Textbox(label="Your Message"),  # User input
        gr.State([])  # Single state to handle persistent data
    ],
    outputs=[
        gr.Chatbot(label="Chat History", type="messages"),  # Set type='messages' to resolve warning
        gr.State([])  # Single state output
    ],
    title="Doctor Appointment Scheduler"
).launch(share=True)
