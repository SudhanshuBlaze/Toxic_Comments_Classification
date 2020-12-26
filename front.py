import joblib
import spacy
import streamlit as st
nlp=spacy.load("en_core_web_lg")

lr_toxic = joblib.load('toxic.joblib')
lr_stoxic = joblib.load('stoxic.joblib')
lr_obscene = joblib.load('obscene.joblib')
lr_threat = joblib.load('threat.joblib')
lr_insult = joblib.load('insult.joblib')
lr_ihate = joblib.load('ihate.joblib')
tfidf = joblib.load('tfidf.joblib')


def main():
    st.title("Toxicos")
    st.subheader('Identify toxic comments')
    raw_comment = st.text_area('Enter The Comment', 'Type Here...')
    if st.button('Check Toxicity'):
        filtered_comment = text_process(raw_comment)
        vector_comment = tfidf.transform([filtered_comment])
        result = prediction(vector_comment)
        st.success(result)


def prediction(vector_comment):
    mes = "This comment is"
    c = 0
    t_percent = lr_toxic.predict_proba(vector_comment)[0][1]
    if (t_percent*100) >= 50:
        mes += f"{t_percent * 100} % toxic"
        c += 1

    st_percent = lr_stoxic.predict_proba(vector_comment)[0][1]
    if (st_percent*100) >= 50:
        mes += f", {st_percent * 100} % severe toxic"
        c += 1

    o_percent = lr_obscene.predict_proba(vector_comment)[0][1]
    if (o_percent*100) >= 50:
        mes += f", {o_percent * 100} % obscene"
        c += 1

    th_percent = lr_threat.predict_proba(vector_comment)[0][1]
    if (th_percent*100) >= 50:
        mes += f", {th_percent * 100} % threatening"
        c += 1

    in_percent = lr_insult.predict_proba(vector_comment)[0][1]
    if (in_percent*100) >= 50:
        mes += f", {in_percent * 100} % insulting "
        c += 1

    ih_percent = lr_ihate.predict_proba(vector_comment)[0][1]
    if (ih_percent*100) >= 50:
        mes += f"and {ih_percent * 100} % attack on identity "
        c += 1

    if c == 0:
        return "This comment is clean"

    return mes


def text_process(raw_comment):
    doc=nlp(raw_comment)
    final=[token.lemma_ for token in doc if token.is_stop== False and token.text.isalpha()== True ]
    return " ".join(final)


if __name__ == '__main__':
    main()