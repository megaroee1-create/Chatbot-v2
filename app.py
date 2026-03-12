import streamlit as st
import torch
import json
import os
import gdown
from model import Voc, EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, normalizeString, evaluate

# --- 1. DOWNLOAD WEIGHTS FROM DRIVE ---
def download_weights():
    file_id = '1eyuN-UhJuldevKJrzkCPqbUbQFHaoNRK'  # <--- REPLACE THIS WITH YOUR ACTUAL FILE_ID
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'weights.tar'
    
    if not os.path.exists(output):
        with st.spinner("Downloading model weights from Google Drive... This may take a minute."):
            gdown.download(url, output, quiet=False)

# Call the download function before anything else
download_weights()

device = torch.device("cpu")

@st.cache_resource
def load_chatbot():
    # 2. LOAD HYPERPARAMETERS
    with open('hyperparameters.json', 'r') as f:
        params = json.load(f)
    
    # 3. REBUILD VOCABULARY
    voc = Voc(params['model_name'])
    for word in params['words']:
        voc.addWord(word)

    # 4. INITIALIZE MODEL COMPONENTS
    embedding = torch.nn.Embedding(voc.num_words, params['hidden_size'])
    
    encoder = EncoderRNN(params['hidden_size'], embedding, params['encoder_n_layers'], params['dropout'])
    decoder = LuongAttnDecoderRNN(params['attn_model'], embedding, params['hidden_size'], 
                                  voc.num_words, params['decoder_n_layers'], params['dropout'])

    # 5. LOAD WEIGHTS
    checkpoint = torch.load('weights.tar', map_location=device)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # 6. INITIALIZE SEARCHER
    searcher = GreedySearchDecoder(encoder, decoder)
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    return voc, encoder, decoder, searcher

# Initialize the model
voc, encoder, decoder, searcher = load_chatbot()

# --- Streamlit Chat Interface (Same as before) ---
st.title("🤖 My PyTorch Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        norm_prompt = normalizeString(prompt)
        response_words = evaluate(encoder, decoder, searcher, voc, norm_prompt, device)
        output = ' '.join([w for w in response_words if w not in ['EOS', 'PAD', 'SOS', 'UNK']])
    
    with st.chat_message("assistant"):
        st.markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})