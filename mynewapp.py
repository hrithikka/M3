import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# load the patent data
df = pd.read_csv("train_data.csv")

# get the unique application numbers
app_numbers = df['patent_number'].unique().tolist()

# define a function to generate the patentability score
# define a function to generate the patentability score
def generate_score(application_filing_number, abstract, claims):
    # retrieve the patent sections using the filing number
    patent_data = df[df['patent_number'] == application_filing_number]
    inputs = tokenizer(patent_data['abstract'].iloc[0], patent_data['claims'].iloc[0], truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    score = outputs.logits[0][1].item()
    # return the patentability score
    return score


# define the Streamlit app interface
st.title("Patentability Score App")

# add a dropdown menu to select the application filing number
application_filing_number = st.selectbox("Select the patent number:", options=app_numbers)

# get the patent sections using the selected filing number
patent_data = df[df['patent_number'] == application_filing_number]

# display the patent sections in text boxes
abstract = st.text_area("Abstract:", value=patent_data['abstract'].iloc[0], height=200)
claims = st.text_area("Claims:", value=patent_data['claims'].iloc[0], height=200)

# add a button to generate the patentability score
if st.button("Generate Score"):
    score = generate_score(application_filing_number, abstract, claims)
    st.write(f"The patentability score is {score:.2f}.")

