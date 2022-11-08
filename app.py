import joblib
import spacy
import streamlit as st
import subprocess

# loads spacy small model
nlp=spacy.load("en_core_web_sm")

lr_toxic = joblib.load('./models/toxic.joblib')
lr_stoxic = joblib.load('./models/stoxic.joblib')
lr_obscene = joblib.load('./models/obscene.joblib')
lr_threat = joblib.load('./models/threat.joblib')
lr_insult = joblib.load('./models/insult.joblib')
lr_ihate = joblib.load('./models/ihate.joblib')
tfidf = joblib.load('./models/tfidf.joblib')

def prediction(vector_comment):
    mes = "This comment is: "
    flag=False
    
    if lr_toxic.predict(vector_comment)[0]==1:
        mes += "toxic"
        flag=True

    if lr_stoxic.predict(vector_comment)[0] ==1:
        mes += ", severe toxic"
        flag=True

    if lr_obscene.predict(vector_comment)[0]==1:
        mes += ", obscene"
        flag=True

    if lr_threat.predict(vector_comment)[0]==1:
        mes += ", threatening"
        flag=True

    if lr_insult.predict(vector_comment)[0]==1:
        mes += ", insulting "
        flag=True

    if lr_ihate.predict(vector_comment)[0]==1:
        mes += ", attack on identity "
        flag=True

    if flag!=True:
        return "This comment is clean",flag

    return mes,flag


def text_process(raw_comment):
    doc=nlp(raw_comment)
    final=[token.lemma_ for token in doc if token.is_stop== False and token.text.isalpha()== True ]
    return " ".join(final)

def main():
	st.title("Toxicos")
	st.subheader('Identify toxic comments')
	raw_comment = st.text_area('Enter The Comment', 'Type Here...')
	if st.button('Check Toxicity'):
		filtered_comment = text_process(raw_comment)
		vector_comment = tfidf.transform([filtered_comment])
		result, flag = prediction(vector_comment)
		if flag==False:
			st.success(result)
		else:
			st.warning(result)

if __name__ == '__main__':
    main()