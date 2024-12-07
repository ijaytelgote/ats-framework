import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import logging
import threading

import fitz  # PyMuPDF
from flask import Flask, jsonify, request
from apps import (all_other, education_master, finale, last_score,
                  logic_actionable_words, logic_similarity_matching2,
                  main_score, master_score, resume_parsing_2, to_check_exp)

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
country = 'India'

def ensure_all_scores(score_dict, required_keys):
    for key in required_keys:
        if key not in score_dict:
            score_dict[key] = 0

def extract_text_from_pdf(pdf_file):
    try:
        pdf_content = pdf_file.read()
        pdf_text = ""
        with fitz.open(stream=pdf_content, filetype="pdf") as pdf:
            for page in pdf:
                pdf_text += page.get_text()
        return pdf_text
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return None

def run_thread(target, *args):
    try:
        thread = threading.Thread(target=target, args=args)
        thread.start()
        thread.join()
    except Exception as e:
        logging.error(f"Error in {target.__name__}: {e}")

def convert_tuple_to_int(t):
    return t[0] if isinstance(t, tuple) and len(t) == 1 else None

def start(pdf_path, text_input):
    resume = extract_text_from_pdf(pdf_path)
    jd = text_input

    if resume and jd:
        run_thread(education_master, resume, master_score, country)
        run_thread(finale, resume, master_score)
        run_thread(resume_parsing_2, resume, master_score)
        run_thread(to_check_exp, resume, jd, main_score)
        run_thread(logic_actionable_words, resume, master_score)
        run_thread(logic_similarity_matching2, resume, jd, master_score)
        run_thread(all_other, master_score, pdf_path)

        required_master_keys = [
            'score_education_detection_', 'score_other',
            'similarity_matching_score', 'Action_score',
            'matrix_score'
        ]
        required_main_keys = ['exp_match', 'Parsing_score']

        ensure_all_scores(master_score, required_master_keys)
        ensure_all_scores(main_score, required_main_keys)

        all_score = [
            master_score['score_education_detection_'],
            master_score['score_other'],
            master_score['similarity_matching_score'],
            master_score['Action_score'],
            master_score['matrix_score']
        ]
        if_resume = master_score['Parsing_score']

        work_exp_matches = main_score['exp_match']
        scoring = last_score(all_score, work_exp_matches)
        print(">>>>>>>>>",if_resume)
        if if_resume == 1:
            logging.info("Resume parsed successfully")
            if isinstance(scoring, str):
                current_value = float(scoring.strip('%'))
            else:
                current_value = float(scoring)
        else:
            logging.info(f"Resume not parsed successfully: {if_resume}")
            current_value = 0

        return current_value

import logging

import fitz  # PyMuPDF
from flask import Flask, jsonify, request


@app.route('/calculate_score', methods=['POST'])
def calculate_score():
    # Retrieve the uploaded file
    pdf_file = request.files.get('pdf_file')
    jd = request.form.get('jd')  # Use form-data for JD

    if not pdf_file or not jd:
        return jsonify({"error": "PDF file and Job Description are required"}), 400

    # Process the file and JD
    score = start(pdf_file, jd)

    # Return the result
    return jsonify({"score": score})


if __name__ == '__main__':
    
    app.run(debug=True, threaded=False,use_reloader=False)

