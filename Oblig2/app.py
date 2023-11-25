import streamlit as st
from transformers import T5ForConditionalGeneration, T5TokenizerFast

def paraphrase(text):
    model_dir = './models/outputs_4'
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5TokenizerFast.from_pretrained(model_dir)
    text = 'paraphrase: ' + text
    tokenized_input = tokenizer(text, return_tensors="pt")
    encoded_output = model.generate(**tokenized_input, max_length=1000)[0]
    paraphrased_text = tokenizer.decode(encoded_output, skip_special_tokens=True)
    return paraphrased_text

st.set_page_config(page_title='✏️🌪️ Paraphrasing App', page_icon='🌪️', layout='centered', initial_sidebar_state='auto')
st.title('✏️🌪️ Paraphrasing App')

txt_input = st.text_area('Enter your text', '', height=50)

result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Paraphrasing...'):
            response = paraphrase(txt_input)
            result.append(response)

if len(result):
    st.info(response)
