---
title: ATS Scoring App
emoji: 📝
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# ATS Scoring App

Welcome to the ATS Scoring App. This app evaluates resumes based on their alignment with a job description (JD) using various scoring criteria.

## Features

- **Keyword-Based Similarity:** Measures how well the resume matches the job description based on keywords.
- **Positive/Negative Words:** Assesses the impact of positive and negative words in the resume.
- **Resume Format:** Checks the resume formatting for compliance with best practices.
- **Content Analysis:** Detects images, tables, graphs, and charts in the resume.
- **JD Presence:** Identifies if the job description is included in the resume.
- **Word Matrix Calculation:** Analyzes word matrices using the XYZ formula.
- **Cosine Similarity:** Computes similarity scores using cosine similarity.
- **Experience Match:** Compares the user's experience with the job description requirements.
- **Education Match:** Evaluates the educational qualifications listed in the resume.
- **Resume Parsability:** Determines if the resume is in a format that can be parsed effectively.

## How to Use

1. **Access the App:**
   Visit the [Hugging Face Spaces page for ATS Scoring App](https://huggingface.co/spaces/ijtelgote/ATScanner) to interact with the application.

2. **Upload Your Data:**
   - Upload the job description (JD) and resume files through the provided interface.

3. **Get Scores:**
   - The application will process the inputs and provide scores based on the evaluation criteria.
   
## Configuration

The app allows for customization through the `config.json` file. Here’s how you can configure it:

## Features in Development

- **Customizable Parameters:** Future updates may include options to adjust scoring parameters and keyword lists.
- **Advanced Analytics:** Plans to add more detailed analytics and reports.

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please reach out to [ijaytelgote@gmail.com](mailto:ijaytelgote@gmail.com).

