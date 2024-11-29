import threading

import fitz  # PyMuPDF
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from main import (all_other, education_master, finale, last_score,
                  logic_actionable_words, logic_similarity_matching2,
                  main_score, master_score, resume_parsing_2, to_check_exp)

# Define the country variable if needed
country = 'India'

# Set the title of the app
st.title('PDF and Job Description Input')

# File uploader for PDF
uploaded_pdf = st.file_uploader("Upload your Resume", type="pdf")

# Text input for Job Description
text_input = st.text_input("Enter the Job Description")

# Function to ensure all scores are present in the dictionary
def ensure_all_scores(score_dict, required_keys):
    for key in required_keys:
        if key not in score_dict:
            score_dict[key] = 0

# Extract text from the uploaded PDF
def extract_text_from_pdf(file):
    if file is None:
        st.error("No file uploaded")
        return None
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Function to run threads with error handling
def run_thread(target, *args):
    try:
        thread = threading.Thread(target=target, args=args)
        thread.start()
        thread.join()
    except Exception as e:
        st.error(f"Error in {target.__name__}: {e}")

# Process Scan button action
if st.button('Scan'):
    if uploaded_pdf and text_input:
        # Extract data
        resume = extract_text_from_pdf(uploaded_pdf)
        jd = text_input

        if resume and jd:
            # Run all threads with error handling
            run_thread(education_master, resume, master_score, country)
            run_thread(finale, resume, master_score)
            run_thread(resume_parsing_2, resume, master_score)
            run_thread(to_check_exp, resume, jd, main_score)
            run_thread(logic_actionable_words, resume, master_score)
            run_thread(logic_similarity_matching2, resume, jd, master_score)
            run_thread(all_other, master_score, uploaded_pdf)

            # Ensure all expected score keys are present
            required_master_keys = [
                'score_education_detection_', 'score_other', 
                'similarity_matching_score', 'Action_score'
                , 'matrix_score'
            ]
            required_main_keys = ['exp_match','Parsing_score']
            
            ensure_all_scores(master_score, required_master_keys)
            ensure_all_scores(main_score, required_main_keys)

            # Collect scores
            all_score = [
                master_score['score_education_detection_'],
                master_score['score_other'],
                master_score['similarity_matching_score'],
                master_score['Action_score'],
                master_score['matrix_score']
            ]
            if_resume=master_score['Parsing_score'],

            work_exp_matches = main_score['exp_match']
            scoring = last_score(all_score, work_exp_matches)

            # Ensure scoring is numeric
            if if_resume==1:
                if isinstance(scoring,str):
                    current_value = float(scoring.strip('%'))
                else:
                    current_value = float(scoring)
            else:
                current_value=0

            st.subheader("Total Score")

            # Plotly gauge chart code
            plot_bgcolor = 'rgba(0,0,0,0)'
            plot_bgcolor = 'rgba(0,0,0,0)'
            quadrant_colors = [
                plot_bgcolor, "#2bad4e", "#85e043",
                "#eff229", "#f2a529", "#f25829"
            ]
            quadrant_text = [
                "", "<b>Very high</b>", "<b>High</b>",
                "<b>Medium</b>", "<b>Low</b>", "<b>Very low</b>"
            ]
            n_quadrants = len(quadrant_colors) - 1

            min_value = 0
            max_value = 100
            hand_length = 0.25
            hand_angle = np.pi * (1 - (current_value - min_value) / (max_value - min_value))

            fig = go.Figure(
                data=[
                    go.Pie(
                        values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                        rotation=90,
                        hole=0.5,
                        marker_colors=quadrant_colors,
                        text=quadrant_text,
                        textinfo="text",
                        hoverinfo="skip",
                    ),
                ],
                layout=go.Layout(
                    showlegend=False,
                    margin=dict(b=20, t=20, l=20, r=20),
                    width=500,
                    height=500,
                    paper_bgcolor=plot_bgcolor,
                    annotations=[
                        go.layout.Annotation(
                            text=f"<b>Score:</b><br>{current_value}%",
                            x=0.5, xanchor="center", xref="paper",
                            y=0.35, yanchor="bottom", yref="paper",
                            showarrow=False,
                            font=dict(size=14, color="#ffffff")
                        )
                    ],
                    shapes=[
                        go.layout.Shape(
                            type="circle",
                            x0=0.48, x1=0.52,
                            y0=0.48, y1=0.52,
                            fillcolor="#ffffff",
                            line_color="#ffffff",
                        ),
                        go.layout.Shape(
                            type="line",
                            x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                            y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                            line=dict(color="#ffffff", width=4)
                        )
                    ]
                )
            )

            # Display the Plotly chart in Streamlit
            st.plotly_chart(fig)

            # Log all scores
            st.text("Detailed Scores:")
            st.json(master_score)
            st.json(main_score)

        else:
            st.error("Failed to process the inputs. Please try again.")
    else:
        st.error("Please upload both the Resume and Job Description.")
