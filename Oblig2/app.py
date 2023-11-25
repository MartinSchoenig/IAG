import streamlit as st
from transformers import T5ForConditionalGeneration, T5TokenizerFast

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

st.set_page_config(page_title='✏️🌪️ Paraphrasing App', page_icon='🌪️', layout='centered', initial_sidebar_state='auto')
st.title('✏️🌪️ Paraphrasing App')

txt_input = st.text_area('Enter your text', '', height=50)

do_sampling = st.checkbox('Enable Sampling', value=False)
num_samples = 1
if do_sampling:    
    num_samples = st.number_input('Number of Samples', min_value=1, max_value=10, value=1)

result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Paraphrasing...'):
            response = paraphrase(txt_input, do_sampling, num_samples)
            result.append(response)

if len(result):
    st.subheader('Paraphrased Text:')
    if do_sampling:
        for i, paraphrase_text in enumerate(result[0]):
            st.write(f"Sample {i + 1}: {paraphrase_text}")
    else:
        st.write(result[0])
