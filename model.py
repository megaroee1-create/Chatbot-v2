import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
import re
from collections import OrderedDict

# --- 2. Define Global Constants (must match training) ---
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unknown word token
MAX_LENGTH = 10 # Maximum sentence length to consider (must match training)
# USE_CUDA and device will typically be defined in the main application
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Voc Class Definition ---
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
            UNK_token: "UNK",
        }
        self.num_words = 4  # Count default tokens

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        # print('keep_words {} / {} = {:.4f}'.format(
        #     len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        # ))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
            UNK_token: "UNK",
        }
        self.num_words = 4

        for word in keep_words:
            self.addWord(word)

# --- 4. Data Preprocessing Helper Functions ---
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    if not s or s is None:
        return ""
    s = str(s).lower().strip()
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip() # Removed the redundant strip/sub chain
    return s

def indexesFromSentence(voc, sentence):
    return [voc.word2index.get(word, UNK_token) for word in sentence.split(' ')] + [EOS_token]

# --- 5. Model Class Definitions (Encoder, Attention, Decoder, Searcher) ---
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(f"Invalid attention method: {self.method}")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        if self.method == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def scoreDot(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def scoreGeneral(self, hidden, encoder_output):
        return torch.sum(self.W(encoder_output) * hidden, dim=2)

    def scoreConcat(self, hidden, encoder_output):
        energy = torch.tanh(self.W(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), dim=2)))
        energy = energy.tanh()
        return torch.sum(self.v(energy), dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'dot':
            attn_energies = self.scoreDot(hidden, encoder_outputs)
        elif self.method == 'general':
            attn_energies = self.scoreGeneral(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.scoreConcat(hidden, encoder_outputs)

        attn_energies = attn_energies.t()
        attn_weights = F.softmax(attn_energies, dim=1)
        return attn_weights.unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super().__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)

        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        # Handles bidirectional encoder output for unidirectional decoder
        encoder_n_layers_actual = self.encoder.n_layers
        forward_hidden_states = encoder_hidden[0:encoder_n_layers_actual]
        backward_hidden_states = encoder_hidden[encoder_n_layers_actual:2*encoder_n_layers_actual]
        combined_encoder_hidden = forward_hidden_states + backward_hidden_states

        # Assuming decoder_n_layers is accessible or passed. For now, matching encoder's layers if smaller.
        # This needs to be carefully handled during deployment if decoder_n_layers is different.
        decoder_n_layers = self.decoder.n_layers # Assuming decoder has n_layers attribute

        if decoder_n_layers <= encoder_n_layers_actual:
            decoder_hidden = combined_encoder_hidden[:decoder_n_layers]
        else:
            last_combined_layer = combined_encoder_hidden[-1].unsqueeze(0)
            num_extra_layers = decoder_n_layers - encoder_n_layers_actual
            repeated_extra_layers = last_combined_layer.repeat(num_extra_layers, 1, 1)
            decoder_hidden = torch.cat((combined_encoder_hidden, repeated_extra_layers), dim=0)

        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=input_seq.device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=input_seq.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=input_seq.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

# --- 6. Evaluation Functions ---
def evaluate(encoder, decoder, searcher, voc, sentence, device, max_length=MAX_LENGTH):
    """
    Evaluate a single input sentence using greedy decoding.
    Out-of-vocabulary words are mapped to UNK_token.
    """
    # words -> indexes (UNK-safe)
    indexes = [
        voc.word2index.get(word, UNK_token)
        for word in sentence.split(' ')
    ] + [EOS_token]

    indexes_batch = [indexes]

    # Create lengths tensor
    lengths = torch.tensor([len(indexes)])

    # Prepare input batch (max_length, batch_size=1)
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")

    # Decode sentence
    tokens, scores = searcher(input_batch, lengths, max_length)

    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words
