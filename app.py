import streamlit as st
import torch
import json
import os
import gdown
from model import Voc, EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, normalizeString, evaluate

# Weights are too heavy to be saved on github, so instead they are uploaded from drive when the code is run:
def download_weights():
    file_id = '1eyuN-UhJuldevKJrzkCPqbUbQFHaoNRK'  #file path to reach file from drive
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'weights.tar'

    #Show a loading screen while downloading weights
    if not os.path.exists(output):
        with st.spinner("Downloading model weights from Google Drive... This may take a minute."): 
            gdown.download(url, output, quiet=False)

# Call the download function before anything else
download_weights()

device = torch.device("cpu") #connect device to runtime, cpu is enough for only forward passing messages

@st.cache_resource #save everything in cache memory rather than load it all every GUI interaction
def load_chatbot():
    # Load hyperparameters
    with open('hyperparameters.json', 'r') as f:
        params = json.load(f)
    
    # Build vocabulary
    voc = Voc(params['model_name'])
    for word in params['words']:
        voc.addWord(word)

    # Initialize layers
    embedding = torch.nn.Embedding(voc.num_words, params['hidden_size'])
    encoder = EncoderRNN(params['hidden_size'], embedding, params['encoder_n_layers'], params['dropout'])
    decoder = LuongAttnDecoderRNN(params['attn_model'], embedding, params['hidden_size'], 
                                  voc.num_words, params['decoder_n_layers'], params['dropout'])

    # Load weights
    checkpoint = torch.load('weights.tar', map_location=device)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # Initialize searcher
    #searcher = GreedySearchDecoder(encoder, decoder)
    searcher = BeamSearchDecoder(encoder, decoder, 5)
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    return voc, encoder, decoder, searcher

# Initialize the model
voc, encoder, decoder, searcher = load_chatbot()

# Streamlit Chat Interface
st.title("SLM ChatBot :)")

if "messages" not in st.session_state: # Create a messages list
    st.session_state.messages = []
    
# Show previouss messages 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Display input message
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Show loading screen while refining text for input model and generating reponse
    with st.spinner("Thinking..."):
        norm_prompt = normalizeString(prompt)
        response_words = evaluate(encoder, decoder, searcher, voc, norm_prompt, device)
        output = ' '.join([w for w in response_words if w not in ['EOS', 'PAD', 'SOS', 'UNK']])
    # Display output
    with st.chat_message("assistant"):
        st.markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})
