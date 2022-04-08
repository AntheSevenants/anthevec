import numpy as np
from .friendly_tokenizer import FriendlyTokenizer

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
        # Save word pieces
        self.word_pieces = tokenized_outputs["word_pieces"]

        # Input ids need to be saved in case we want to get the raw token embeddings
        self.input_ids = tokenized_outputs["tokenized_sentence"]["input_ids"]

        # Pre-save the token embeddings (maybe we need them at some point)
        embedding_matrix = model.get_input_embeddings()
        self.token_embeddings = embedding_matrix(self.input_ids).detach().numpy()

        # Compute the hidden states
        outputs = model(self.input_ids)
        # Save these hidden states separately
        self.hidden_states = outputs.hidden_states
        self.attentions = outputs.attentions
    
    # Get a hidden state for a specific word (by word index)
    # sentence_index: int, index for a sentence
    # word_index: int, spaCy index for a word!
    # layers: for which layers should we retrieve the hidden states?
    def get_hidden_state(self, sentence_index, word_index, layers, heads=None):
        # You can ask for hidden states over multiple layers
        if not type(layers) == list:
            raise Exception("Layers arguments should be a list")
            
        if 0 in layers and heads is not None:
            raise Exception("Attention is not available for embedding layer 0. " +
                            "Please excluse layer 0 from your layer selection if " +
                            "you want to use attention-based weighting.")
        
        # Compute layer vectors a specific word_index
        # We save all word vectors of each layer in the following list:
        layer_vectors = [] 
        
        # Iterate over all layer indices which were given
        for layer_index in layers:
            # A word can consist of multiple word pieces.
            word_piece_vectors = [] # we store the word piece vectors here
                                        
            # For each of the word pieces of which this word consists...
            for wordpiece_index in self.correspondence[sentence_index][word_index]:
                # ...we retrieve the vector for this word piece
                word_piece_vector = self.get_word_piece_vector(sentence_index, wordpiece_index, layer_index)
                # ...and add it to the list of word pieces for this word
                word_piece_vectors.append(word_piece_vector)
                          
            # We turn the list of word pieces into a two-dimensional matrix...
            word_piece_vectors = np.array(word_piece_vectors)
            
            # These weights will be used when averaging the word piece vectors
            # If attention is not defined, we don't use any custom weights when averaging
            weights = None
            
            # If the attention heads are defined, it means we don't just take the regular average
            # to piece together a vector for this word.
            # Rather, we use the attention values to make a *weighted* average
            if heads is not None:
                weights = self.get_attention_weights(sentence_index,
                                                     word_index,
                                                     layer_index,
                                                     heads,
                                                     word_distribution=True)
            
            # ...and average columnwise, so we get one average vector for this word
            word_vector = np.average(word_piece_vectors, 0, weights=weights)
            # Then, we add the vector to the list of word vectors across layers
            layer_vectors.append(word_vector)
        
        # We turn the list of word vectors across layers into a two-dimensional matrix...
        layer_vectors = np.array(layer_vectors)
        # ...and average columnwise, so we get an average vector over all requested layers
        layer_average = np.average(layer_vectors, 0)
        
        return layer_average

    def get_word_piece_vector(self, sentence_index, wordpiece_index, layer_index):
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

        # All done
        return word_piece_vector

    def get_attention_slice(self, layer_index, sentence_index, head_index, wordpiece_index):
        if layer_index == 0:
            raise Exception("Cannot get attention distribution for embedding layer.")

        # How are attention weights structured?
        # attentions = tuple
        # every item in the tuple = attention for one layer => length of tuple = 12
        # /!\ attention (haha): hidden_states[1] has attentions[0]
        # hidden_states[0] has no attention (this is the embedding layer)
        # each tuple item has the following shape:
        # dim 1: sentence index
        # dim 2: attention head index
        # dim 3: word piece index
        # dim 4: word piece index (for the attention distribution)
                
        # This "attention slice" is the attention range over ALL word pieces, FOR THIS PIECE ONLY
        # It also only applies to THIS HEAD ONLY, and FOR THIS LAYER ONLY
        attention_slice = self.attentions[layer_index - 1][sentence_index][head_index][wordpiece_index].tolist()

        return attention_slice
    
    def get_attention_weights(self, sentence_index, word_index, layer_index, heads, word_distribution=False):
        # We store the attention weights across heads here
        heads_attention_weights = [] 
        for head_index in heads:
            # We store the attention weights for word pieces here
            wordpieces_attention_weights = []
            for wordpiece_index in self.correspondence[sentence_index][word_index]:
                attention_slice = self.get_attention_slice(layer_index, sentence_index, head_index, wordpiece_index)

                # True if we are only interested in attention among word pieces belonging to this word
                if word_distribution:
                    # We are only interested in the attention range between the wordpieces of our word
                    slice_begin = self.correspondence[sentence_index][word_index][0]
                    slice_end = self.correspondence[sentence_index][word_index][-1] + 1 # slice excludes the final index
                    attention_slice = attention_slice[slice_begin:slice_end]
                        
                # We add the attention slice to the list 
                wordpieces_attention_weights.append(attention_slice)
                    
            # We now have all attention slices for this head for this word
            # For each word piece belonging to this word, we know how important it finds itself, and the other pieces
                    
            # We turn the list of attention weights for word pieces into a two-dimensional matrix...
            wordpieces_attention_weights = np.array(wordpieces_attention_weights)
            # ...and average columnwise, so we get one average attention weight distribution for this head
            head_attention_weights = np.average(wordpieces_attention_weights, 0)
            # Now, make the weights sum to one
            head_attention_weights = head_attention_weights / head_attention_weights.sum(axis=0, keepdims=True)
            # Finally, add these weights to the list of attention heads requested
            heads_attention_weights.append(head_attention_weights)
                
        # We now have all attention weights for all heads for this word
        # For each head, we now know the attention distribution among the word pieces
                
        # We turn the list of attention weights for all heads into a two-dimensional matrix...
        heads_attention_weights = np.array(heads_attention_weights)
        # ...and average columnwise, so we get one average attention distribution across all heads for our word pieces
        # We don't weight the attention heads (yet?), so we just average the different heads
        heads_attention_weights = np.average(heads_attention_weights, 0)
        # Now, make the weights sum to one
        heads_attention_weights = heads_attention_weights / heads_attention_weights.sum(axis=0, keepdims=True)
                
        # All done!
        weights = heads_attention_weights.tolist()
        
        return weights

    def get_token_embedding(self, sentence_index, word_index):
        word_piece_vectors = []

        # For each of the word pieces of which this word consists...
        for wordpiece_index in self.correspondence[sentence_index][word_index]:
            # Define the token embedding...
            word_piece_vector = self.token_embeddings[sentence_index][wordpiece_index]
            # ...and add it to the list of word pieces for this word
            word_piece_vectors.append(word_piece_vector)
              
        # We turn the list of word pieces into a two-dimensional matrix...
        word_piece_vectors = np.array(word_piece_vectors)
        # ...and average columnwise, so we get one average vector for this word
        word_vector = np.average(word_piece_vectors, 0)

        return word_vector