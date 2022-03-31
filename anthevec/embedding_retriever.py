import numpy as np
from anthevec.friendly_tokenizer import FriendlyTokenizer

# EmbeddingRetriever holds all embeddings for a given sentence
class EmbeddingRetriever:
    def __init__(self, model, tokenizer, nlp, input_sentences):
        # Initialise the tokeniser
        self.friendly_tokenizer = FriendlyTokenizer(tokenizer, nlp)
        # Tokenise the input sentence
        tokenized_outputs = self.friendly_tokenizer.tokenize(input_sentences)
        #rint(tokenized_outputs)
        # Save the correspondence dict
        self.correspondence = tokenized_outputs["correspondence"]
        # Save tokens
        self.tokens = tokenized_outputs["spacy_tokens"]
        
        # Compute the hidden states
        outputs = model(tokenized_outputs["tokenized_sentence"]["input_ids"])
        # Save these hidden states separately
        self.hidden_states = outputs.hidden_states
    
    # Get a hidden state for a specific word (by word index)
    # sentence_index: int, index for a sentence
    # word_index: int, spaCy index for a word!
    # layers: for which layers should we retrieve the hidden states?
    def get_hidden_state(self, sentence_index, word_index, layers):
        # You can ask for hidden states over multiple layers
        if not type(layers) == list:
            raise Exception("Layers arguments should be a list")
        
        # Compute layer vectors a specific word_index
        # We save all word vectors of each layer in the following list:
        layer_vectors = [] 
        
        # Iterate over all layer indices which were given
        for layer_index in layers:
            # A word can consist of multiple word pieces.
            word_piece_vectors = [] # we store the word piece vectors here
                                        
            # For each of the word pieces of which this word consists...
            for wordpiece_index in self.correspondence[sentence_index][word_index]:
                # Define the hidden state, and detach it so it becomes a numpy array
                hidden = self.hidden_states[layer_index].detach().numpy()
                # The hidden state is actually a *tensor* (three dimensions), so we need to reshape
                # it so it becomes a two-dimensional matrix. The first dimension is the sentence index,
                # so no actual information is lost.                
                flat_hidden_state = hidden[sentence_index].reshape((hidden.shape[1], hidden.shape[2]))
                
                # We slice the matrix and only get the row that corresponds to the current word piece
                word_piece_vector = flat_hidden_state[wordpiece_index]
                # Then, we convert it into a simple vector...
                word_piece_vector = word_piece_vector.reshape(hidden.shape[2])
                # ...and add it to the list of word pieces for this word
                word_piece_vectors.append(word_piece_vector)
                          
            # We turn the list of word pieces into a two-dimensional matrix...
            word_piece_vectors = np.array(word_piece_vectors)
            # ...and average columnwise, so we get one average vector for this word
            word_vector = np.average(word_piece_vectors, 0)
            # Then, we add the vector to the list of word vectors across layers
            layer_vectors.append(word_vector)
        
        # We turn the list of word vectors across layers into a two-dimensional matrix...
        layer_vectors = np.array(layer_vectors)
        # ...and average columnwise, so we get an average vector over all requested layers
        layer_average = np.average(layer_vectors, 0)
        
        return layer_average