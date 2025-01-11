import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import fitz  # The `calculate_percentage` function is not defined in the provided code snippet. It seems that it is not being used or imported in the current implementation. If you need assistance with a specific function or have any other questions, feel free to ask!
import numpy as np
import pandas as pd
import pdfplumber
import pendulum
import requests
import tensorflow as tf
import tensorflow_hub as hub
import torch
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (BertModel, BertTokenizer,
                          TFBertForSequenceClassification)

# Set the logging level to WARNING to suppress DEBUG and INFO logs
logging.basicConfig(level=logging.WARNING)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import threading

logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set the logging level to WARNING to suppress DEBUG and INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("groq._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

load_dotenv()

def pdf_to_text(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

json_path='country_data.json'


bert_tokenizer = BertTokenizer.from_pretrained('output_directory2')
bert_model = TFBertForSequenceClassification.from_pretrained('Model2')



class MasterOther:
    def __init__(self):
        self.backlink_score = 0
        self.font_size_score = 0
        self.font_name_score = 0
        self.image_score = 0
        self.table_score = 0
        self.page_count_score = 0

    def normalize_value(self, value, min_value, max_value):
        if value < min_value:
            return 0
        return (value - min_value) / (max_value - min_value)

    def extract_pdf_fonts_and_sizes(self, pdf_file):
        # Reset the file pointer to the start of the stream
        pdf_file.seek(0)

        # Open the PDF using PyMuPDF
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        font_sizes = set()

        # Iterate through the pages
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.add(span["size"])

        doc.close()
        return font_sizes

    def extract_pdf_fonts_and_sizes_score(self, pdf_file):
        try:
            font_sizes = self.extract_pdf_fonts_and_sizes(pdf_file)
        except ValueError as e:
            print(e)
            self.font_size_score = 0
            return

        score = 20
        max_score = 20
        min_score = 0

        for size in font_sizes:
            if size > 20.0 or size < 5.0:
                score = 0
                print('Tailor the font size accordingly')
                break
        self.font_size_score = self.normalize_value(score, min_score, max_score)

    def check_backlinks(self, pdf_file):
        try:
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        except Exception as e:
            print(f"Error opening PDF for backlinks: {e}")
            self.backlink_score = 0
            return

        score = 10
        max_score = 10
        min_score = 0
        page = doc.load_page(0)
        links = page.get_links()
        if links:
            score = 0
            print('Resume contains backlinks')
        doc.close()
        self.backlink_score = self.normalize_value(score, min_score, max_score)

    def contains_table(self, pdf_file):
        try:
            pdf_file.seek(0)
            with pdfplumber.open(pdf_file) as pdf:
                score = 10
                max_score = 10
                min_score = 0
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        score -= 5
                        if score == 0:
                            break
                if score < max_score:
                    print('Resume contains tables')
                self.table_score = self.normalize_value(score, min_score, max_score)
        except Exception as e:
            print(f"Error checking tables: {e}")
            self.table_score = 0

    def contains_images(self, pdf_file):
        try:
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        except Exception as e:
            print(f"Error opening PDF for images: {e}")
            self.image_score = 0
            return

        score = 10
        max_score = 10
        min_score = 0
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            if image_list:
                score = 0
                print('Resume contains images')
                break
        doc.close()
        self.image_score = self.normalize_value(score, min_score, max_score)

    def detect_fonts(self, pdf_file):
        try:
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        except Exception as e:
            print(f"Error opening PDF for font detection: {e}")
            return {}

        font_counts = {}

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_name = span["font"]
                            if font_name in font_counts:
                                font_counts[font_name] += 1
                            else:
                                font_counts[font_name] = 1

        doc.close()
        return font_counts

    def tune_font(self, pdf_file):
        try:
            font_counts = self.detect_fonts(pdf_file)
        except Exception as e:
            print(f"Error detecting fonts: {e}")
            self.font_name_score = 0
            return

        score = 100
        max_score = 100
        min_score = 18
        never_use_fonts = ['Comic Sans', 'Futura', 'Lucida Console', 'Bradley Hand ITC', 'Brush Script']
        for font, count in font_counts.items():
            if font in never_use_fonts:
                score -= count * 18
                print(f"{font} is not recommended for resume")
                break
        self.font_name_score = self.normalize_value(score, min_score, max_score)

    def count_pdf_pages_score(self, pdf_file):
        try:
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        except Exception as e:
            print(f"Error opening PDF for page count: {e}")
            self.page_count_score = 0
            return

        num_pages = doc.page_count
        doc.close()
        score = 30
        max_value = 30
        min_value = 7
        if num_pages == 2:
            score -= 13
        elif num_pages > 2:
            print('Resume should not be more than 2 pages')
            score -= 23
        if score < min_value:
            score = 0
        self.page_count_score = self.normalize_value(score, min_value, max_value)

def all_other(master_score, pdf_file):
    master = MasterOther()
    try:
        master.extract_pdf_fonts_and_sizes_score(pdf_file)
        master.check_backlinks(pdf_file)
        master.contains_table(pdf_file)
        master.contains_images(pdf_file)
        master.tune_font(pdf_file)
        master.count_pdf_pages_score(pdf_file)
    except Exception as e:
        print(f"Error in all_other function: {e}")
        master_score['score_other'] = 0
        return

    mean = (
        (master.font_size_score +
         master.table_score +
         master.font_name_score +
         master.backlink_score +
         master.page_count_score +
         master.image_score) / 6
    )
    master_score['score_other'] = mean * 100




############################################################## 
    

def calculate_similarity_use(text1, text2):

    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    embeddings = model([text1, text2])

    similarity_score = cosine_similarity(embeddings)[0, 1]

    return similarity_score

def containment_similarity(text1, text2):
    # Tokenize the texts
    set1 = set(text1.split())
    set2 = set(text2.split())

    # Calculate intersection
    intersection = set1.intersection(set2)

    containment_score = len(intersection) / min(len(set1), len(set2))

    return containment_score

import re


def remove_special_characters(text):
    pattern = r"[.,!()*&⋄:|/^]"
    cleaned_text = re.sub(pattern, "", text)
    tex=cleaned_text.replace('\n','')
    return tex.lower()
def logic_similarity_matching(text1,text2):

  score_encoder=0
  score_containment=0
  text1=remove_special_characters(text1)
  text2=remove_special_characters(text2)
  similarity_score_use = calculate_similarity_use(text1, text2)

  similarity_score = containment_similarity(text1, text2)
  if similarity_score>0.8:
    score_containment+=1

  if similarity_score_use>=0.75:
    score_encoder=1
  return score_encoder == 1 and score_containment == 0

def normalize_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def logic_similarity_matching2(text1,text2,master_score):
    score=10
    max_score=10
    min_score=0
    if logic_similarity_matching(text1,text2)==False:
        score-=10
        print('Resume not tailored according to JD')
        
    master_score['similarity_matching_score']= normalize_value(score, min_score, max_score)
    
#################################################

master_score={}
def get_bert_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()
import joblib


# Function to predict resume scores
def predict_resume_score(new_resumes):
    # Generate BERT embeddings for the new resumes
    regressor = joblib.load('model_filename2.pkl')

    embeddings = get_bert_embeddings(new_resumes)
    X_new = torch.tensor(embeddings, dtype=torch.float32)
    
    # Predict using the trained Random Forest Regressor
    predictions = regressor.predict(X_new)
    
    return predictions
def normalize_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)



def logic_actionable_words(text,master_score):
    score=0
    max_score=100
    min_score=0
    pred_score=predict_resume_score(text)[0]*100
    if int(pred_score)>50:
        score=100
    elif int(pred_score)>=40 and int(pred_score)<=49:
        score=80
    elif int(pred_score)>=30 and int(pred_score)<=39:
        score=60
    elif int(pred_score)>=20 and int(pred_score)<=29:
        print("Either Resume contain some Non Action Keywords or Resume don't has Actionable Keywords")
        score=40
    else:
        score=10
        print("Either Resume contain some Non Action Keywords or Resume don't has Actionable Keywords")
    master_score['Action_score']= normalize_value(score, min_score,max_score)
    
    
    
   ############################################################## 
groq_api_key = os.getenv('API_KEY')


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='llama3-70b-8192'
)



def keep_only_alphanumeric(text):
    pattern = r'[^a-zA-Z0-9]'

    cleaned_text = re.sub(pattern, ' ', text.lower())
    return ' '.join(cleaned_text.split())

def groq(jd):


    system = '''
    1. Act as a Minimum No of Experience required telling person.
    2. The user will provide input as whole Job description, you have to provide minimum no of experience required to apply for the job as its must be clearly mentioned in JD.
    3 . If found something like Freshers can apply or no experience requied, respond with `0.0`.
    3. If you not able find any experience explicitly mentioned in job description, then just respond with `0.0`.
    4. If the provided description is not any kind of JD(if it is something else apart from JD, like artical random pdf, not a jd), then just repond with `False`.
    4. Do not give any introduction about who you are and what you are going to do.
    5. Don't try to give false experience.
    
    5. you will give the no of experience in numbers like `8 years`, not `eight years` or something.
    6. remember this formula "Years= Month no/12" so for 2 months it is 0.17 in the round figure
    7. Always give no of experience in decimal. Let say for 4 years, you should give it 4.0. Similarly for 6 months, you  should give 0.5. Don't add any alphabetic words there.
    '''
    
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | llm
    jd2=keep_only_alphanumeric(jd)
    res = chain.invoke({"text": jd2})
    p = dict(res)
    final_text = ' '.join(p['content'].split())
    return final_text


def parse_date(date_str):
    try:
        parsed_date = pendulum.parse(date_str, strict=False)
        return parsed_date
    except ValueError:
        raise ValueError(f"No valid date format found for '{date_str}'")

def calculate_experience(start_date, end_date):
    duration = end_date.diff(start_date)
    years = duration.years
    months = duration.months
    return years + months / 12

def calculate_total_experience(resume_text):
    # Regular expression to match date ranges with various formats including year-only ranges
    date_range_pattern = re.compile(
        r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\s\'\"`*+,\-–.:/;!@#$%^&(){}\[\]<>_=~`]*\d{2,4}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{4}|\d{4})\s*(?:[-–to ]+)\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\s\'\"`*+,\-–.:/;!@#$%^&(){}\[\]<>_=~`]*\d{2,4}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{4}|\d{4}|\b[Tt]ill\b|\b[Nn]ow\b|\b[Pp]resent\b|\b[Oo]ngoing\b|\b[Cc]ontinue\b|\b[Cc]urrent\b)?'
    )

    date_matches = date_range_pattern.findall(str(resume_text))
        
    total_experience = 0
    
    for start_date_str, end_date_str in date_matches:
        try:
            start_date = parse_date(start_date_str.strip())
            end_date = pendulum.now() if not end_date_str or end_date_str.strip().lower() in ['till', 'now', 'present', 'ongoing', 'continue', 'current'] else parse_date(end_date_str.strip())
                        
            experience = calculate_experience(start_date, end_date)
            
            total_experience += experience
        except ValueError as e:
            print(e)
    
    return round(total_experience, 2)
#calculate_total_experience(resume)



def extract_experience(resume):
    system = f"extract the only Experience section in the resume if contains or the user mentioned in its resume and don't add anything else by your own, if there is no experience present on resume just write 'None', remember not to consider Projects and Extra Curriculum, etc, start your answer with `>`."
    
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | llm
    res = chain.invoke({"text": resume})
    p = dict(res)
    final_text = ' '.join(p['content'].split())
    return final_text

main_score = {}
def to_check_exp(resume: str, jd: str, main_score: dict) -> None:

    try:
        # Extract required experience from the job description
        required_experience = groq(jd)
        
        if required_experience == 'False' or required_experience is None:
            print("Please enter a valid Job Description.")
            main_score['exp_match'] = 0  # Default to no match if JD is invalid
            return
        
        required_experience = float(required_experience)
        print(f"Required Experience: {required_experience} years")

        # Extract and calculate candidate's total experience
        extracted_exp = extract_experience(resume)
        candidate_experience = float(calculate_total_experience(extracted_exp))
        print(f"Candidate Experience: {candidate_experience} years")

        # Compare experience and update main_score
        if candidate_experience >= required_experience:
            print("User experience matches the Job Description.")
            main_score['exp_match'] = 1
        else:
            print("User experience does not match the Job Description.")
            print(f"User Exp: {candidate_experience}, Required Exp: {required_experience}")
            main_score['exp_match'] = 0

    except (ValueError, TypeError) as e:
        print(f"Error while processing: {e}")
        main_score['exp_match'] = 0  # Default to no match on error

####################################################
def extract_skills(text):
    skills_pattern = (
        r"\b(Skill(?:s|z)?|Abilit(?:ies|y|tys)?|Competenc(?:ies|y)|Expertise|Skillset|Technical Skills?|Technical Abilities?|Technological Skills?|TECHNICAL SKILLS?|Technical Expertise)\b"
        r"[\s:\-\n]*"
        r"(.+?)(?=\b(Experience|Experiences|Employment|Work History|Professional Background|Projects|its last|Project Work|Case Studies|Education|Educations|Academic Background|Qualifications|Studies|Soft Skills|Achievements|$))"
    )
    skills_match = re.search(skills_pattern, text, re.DOTALL | re.IGNORECASE)
    if skills_match:
        return skills_match.group(2).strip()
    return None

def extract_education(text):
    education_pattern = (
        r"\b(Education|Educations|Academic Background|Qualifications|Studies|Academic Qualifications|Educational Background|Academic History|Educational History|Education and Training|Educational Qualifications|EDUCATION)\b"
        r"[\s:\-\n]*"
        r"(.+?)(?=\b(Skills?|Abilities?|Competenc(?:ies|y)|Expertise|Skillset|Technical Skills?|Technical Abilities?|Experience|Experiences|Employment|Work History|Professional Background|Projects?|Project Work|Case Studies|its last|Soft Skills|Achievements|$))"
    )
    education_match = re.search(education_pattern, text, re.DOTALL | re.IGNORECASE)
    if education_match:
        return education_match.group(2).strip()
    return None




def parsed(resume1):
    resume1=resume1+'    its last'
    resume1=resume1.replace('\n',' ')

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+\d{2,4}[-.\s]?\d{10}|\d{10}|\d{11})'

    email_match = re.search(email_pattern, resume1)
    phone_match = re.search(phone_pattern, resume1)

    email = email_match.group() if email_match else None
    phone = phone_match.group() if phone_match else None


    skills = extract_skills(resume1)

    experience = extract_experience(resume1)

    education = extract_education(resume1)



    return {
        'Email': email,
        'Phone': phone,
        'Skills': skills.replace('\n','') if skills else None,
        'Experience': experience.replace('\n','') if experience else None,
        'Education': education.replace('\n','') if education else None,
    }

def resume_parsing_2(resume,master_score):
    parsed_resume = parsed(resume)
    if any(value is None for value in parsed_resume.values()):
        print('Resume template is not ATS friendly.')
        master_score['Parsing_score']= 0
    else:
        master_score['Parsing_score']= 1


###################################



import os
import sys


def Get_sentiment(Review, Tokenizer=bert_tokenizer, Model=bert_model, threshold=0.5):
    if not isinstance(Review, list):
        Review = [Review]
    Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(Review,
                                                                            padding=True,
                                                                            truncation=True,
                                                                            max_length=128,
                                                                            return_tensors='tf').values()
    
    # Redirect stdout to suppress progress messages
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')  # Suppress output

    prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask])
    
    # Restore stdout
    sys.stdout.close()
    sys.stdout = original_stdout

    probs = tf.nn.softmax(prediction.logits, axis=1)
    pred_labels = tf.argmax(probs, axis=1)
    pred_probs = probs.numpy().tolist()
    return pred_probs[0][1]



word2vec_model=joblib.load(r'word2vec_res_model.pkl')
model=joblib.load(r'word_matrix_ml_model.pkl')
def get_average_word2vec(words, word2vec_model):
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if not word_vectors:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word_vectors, axis=0)

def another_word2vec(texts):
  X_new = np.array([get_average_word2vec(texts, word2vec_model)])

  new_predictions = model.predict(X_new)
  return  new_predictions

def semi_final(texts):
    min_score = 0
    max_score = 10
    sentiment_score = np.mean([Get_sentiment(text) for text in texts])
    
    if sentiment_score > 0.8:
        word2vec_score = np.mean([another_word2vec(text) for text in texts]) * 100
        
        if 70 <= word2vec_score < 85:
            score = 8
        elif word2vec_score >= 86:
            score = 10
        elif 50 <= word2vec_score < 69:
            score = 6
        elif 30 <= word2vec_score < 49:
            score = 4
        else:
            score = 2
    else:
        print('Resume is not Customized')
        return None

    return normalize_value(score, min_score, max_score)

def normalize_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def process(texts):
    textt=(texts.lower()).replace('\n','').replace('\t','').replace('"','')
    texts2=textt.split('.')
    return [i for  i in texts2 if i!='']

def finale(resume, master_score):
    texts=process(resume)
    score = semi_final(texts)
    if score is not None:
        master_score['matrix_score'] = score
    else:
        master_score['matrix_score'] = 0
        
        
        
#################################################################



def fetch_page(country, page_number):
    path_country=json_path
    with open(path_country, 'r') as file:
        countries_dict = json.load(file)
    
    try:
#        url = f'http://161.111.47.11:80/en/{countries_dict[country]}?page={page_number}'
        url = f'https://www.webometrics.info/en/{countries_dict[country]}?page={page_number}'

        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"Error fetching page {page_number} for {country}: {e}")
        return None

def fetch_all_pages(country, num_pages=2):
    with ThreadPoolExecutor() as executor:
        pages_content = list(executor.map(lambda p: fetch_page(country, p), range(num_pages)))
    return [content for content in pages_content if content]

def parse_pages(pages_content):
    institutions = set()
    for content in pages_content:
        soup = BeautifulSoup(content, 'html.parser')
        rows = soup.select('tbody tr')
        for row in rows:
            name_element = row.select_one('td:nth-of-type(3) a')
            if name_element:
                institution_name = name_element.text.strip().lower()
                institutions.add(institution_name)
    return institutions

def extract_education(text):
    education_pattern = (
        r"\b(Education|Educations|Academic Background|Qualifications|Studies|Academic Qualifications|Educational Background|Academic History|Educational History|Education and Training|Educational Qualifications|EDUCATION)\b"
        r"[\s:\-\n]*"
        r"(.+?)(?=\b(Skills?|Abilities?|Competenc(?:ies|y)|Expertise|Skillset|Technical Skills?|Technical Abilities?|Experience|Experiences|Employment|Work History|Professional Background|Projects?|Project Work|Case Studies|its last|Soft Skills|Achievements|$))"
    )
    education_match = re.search(education_pattern, text, re.DOTALL | re.IGNORECASE)
    return education_match.group(2).strip() if education_match else None

def extract_institutions_from_resume(resume_text):
    pattern = r'[>(><&#%")-:\'\d]'
    res = resume_text.replace('|', '\n')
    cleaned_text = re.sub(pattern, '', res)
    return [re.sub(r'\s+', ' ', inst).strip().lower() for inst in cleaned_text.splitlines() if len(inst.split()) >= 3]

def main(resume_text, country='India'):
    pages_content = fetch_all_pages(country)
    institutions = parse_pages(pages_content)
    
    education_text = extract_education(resume_text)
    if education_text:
        resume_institutions = extract_institutions_from_resume(education_text)
        found_institutions = [name for name in resume_institutions if name in institutions]
        return found_institutions
    return []


def education_master(resume_text, master_score, country):
    score = 0.0
    educ_institutions = main(resume_text, country)
    if educ_institutions:
        if len(educ_institutions) == 1:
            score = 0.5
        elif len(educ_institutions) > 1:
            score = 1.0
    master_score['score_education_detection_'] = score



def normalize_scores(scores):
    # Define the ranges for each score type
    ranges = [
        (0.0, 1.0),   # First score: 0 to 1
        (0.0, 100.0), # Second score: 0 to 100
        (0.0, 1.0),   # Third score: 0 to 1
        (0.0, 1.0),   # Fourth score: 0 to 1
        (0.0, 1.0),   # Fifth score: 0 to 1
        (0.0, 1.0)    # Sixth score: 0 to 1
    ]
    
    # Normalize each score
    normalized_scores = []
    for score, (min_val, max_val) in zip(scores, ranges):
        normalized_score = (score - min_val) / (max_val - min_val) * 100
        normalized_scores.append(normalized_score)
    return normalized_scores

def calculate_percentage(normalized_scores):
    total_score = sum(normalized_scores)
    num_scores = len(normalized_scores)
    
    # Calculate the average normalized score
    average_score = total_score / num_scores
    return average_score



def last_score(all_score, work_exp_matches):
    if work_exp_matches == 1:
        normalized_scores1 = normalize_scores(all_score)
        final_percentage1 = calculate_percentage(normalized_scores1)
        return "{:.2f}".format(final_percentage1)
    else:
        return 0



