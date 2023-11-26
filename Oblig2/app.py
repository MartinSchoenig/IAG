import streamlit as st
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import pandas as pd
import os

def save_feedback(text, paraphrase, csv_filename='feedback.csv'):
    if not os.path.exists(csv_filename):
        df = pd.DataFrame(columns=['Text', 'Paraphrase'])
        df.to_csv(csv_filename, index=False)

    feedback_df = pd.DataFrame({'Text': [text], 'Paraphrase': [paraphrase]})
    df = pd.concat([pd.read_csv(csv_filename), feedback_df], ignore_index=True)
    df.to_csv(csv_filename, index=False)

def paraphrase(text, do_sampling, num_samples):
    model_dir = './models/outputs_4'
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5TokenizerFast.from_pretrained(model_dir)
    
    text = 'paraphrase: ' + text
    
    tokenized_input = tokenizer(text, return_tensors="pt")
    
    if do_sampling:
        encoded_output = model.generate(
            **tokenized_input,
            max_length=1000,
            num_return_sequences=num_samples,
            do_sample=True,
            top_k=120,
            top_p=0.95
        )
        paraphrased_text = [tokenizer.decode(tokenized_output, skip_special_tokens=True) for tokenized_output in encoded_output]
        return paraphrased_text
    else:
        encoded_output = model.generate(**tokenized_input, max_length=1000)[0]
        paraphrased_text = tokenizer.decode(encoded_output, skip_special_tokens=True)
        return paraphrased_text

state = st.session_state
if 'paraphrased_text' not in state:
    state.paraphrased_text = None
if 'feedback' not in state:
    state.feedback = None

st.set_page_config(page_title='✏️🌪️ Paraphrasing', page_icon='🌪️', layout='centered', initial_sidebar_state='auto')
st.title('✏️🌪️ Paraphrasing')

txt_input = st.text_area('Enter your text', '', height=50)

do_sampling = st.checkbox('Enable Sampling', value=False)
num_samples = 1
if do_sampling:    
    num_samples = st.number_input('Number of Samples', min_value=1, max_value=10, value=1)

if st.button('Paraphrase'):
    with st.spinner('Paraphrasing...'):
        response = paraphrase(txt_input, do_sampling, num_samples)
        state.paraphrased_text = response

if state.paraphrased_text:
    st.subheader('Paraphrased Text:')
    if do_sampling:
        for i, paraphrase_text in enumerate(state.paraphrased_text):
            st.success("Sample {}: {}".format(i+1, paraphrase_text))
    else:
        st.success(state.paraphrased_text)

    state.feedback = st.radio("Help us improve our model! Was the resulting (first one if sampled) paraphrase good or bad?", options=['Good', 'Bad'], index=None)
    submit_feedback = st.button('Submit Feedback')
    if submit_feedback:
        if state.feedback == 'Good':
            if do_sampling:
                save_feedback(txt_input, state.paraphrased_text[0])
            else:
                save_feedback(txt_input, state.paraphrased_text)
        
