from .pre_tokenizer import PreTokenizer

class FriendlyTokenizer:
    def __init__(self, tokenizer, nlp):
        self.tokenizer = tokenizer # the BERT wordpiece tokeniser
        self.nlp = nlp # the spaCy tokeniser
        self.tokens = []
    
    def tokenize(self, input_strings):        
        if not type(input_strings) is list:
            raise Exception("Tokenizer input should be a list")

        # It's easiest to just piece together pre-tokenised strings
        # We can pass them as-is with a special parameter, but it leads to issues
        tokenizer_input_strings = []
        if type(input_strings[0]) == list:
            # If the first item is a list, we just join all the list items
            # And then we add these items to the tokenizer input strings
            for sentence_list in input_strings:
                tokenizer_input_strings.append(" ".join(sentence_list))
        elif type(input_strings[0] == str):
            # If the first item is a string, we just use the input as-is
            tokenizer_input_strings = input_strings
        
        # First, tokenize the string as normal using the supplied tokenizer
        tokenized_output = self.tokenizer(tokenizer_input_strings, return_tensors='pt', padding=True)
                
        # Will hold all correspondences per sentence
        correspondence_list = []
        spacy_tokens_list = []
        
        # We look over all sentences (sentence ids)
        for sentence_id in range(len(tokenized_output["input_ids"])):
            # Then, go over each of the elements in the inputs list
            # And find their correspondence
            correspondence = {}
            
            spacy_tokens = []

            if type(input_strings[sentence_id]) == list:
                self.nlp.tokenizer = PreTokenizer(self.nlp.vocab)

            doc = self.nlp(input_strings[sentence_id])

            # Flatten the sentence structure
            for sentence in doc.sents:
                spacy_tokens += sentence
                
            spacy_tokens_list.append(spacy_tokens)

            # We omit the first and last tokens (these are model-specific tokens)
            token_id = 1 # token 0 = CLS token (we skip it)
            for token in tokenized_output["input_ids"][sentence_id][1:-1]:
                #if token_id >= len(spacy_tokens):
                #    break
                
                # We ask the BERT tokenizer to where in the input string this token corresponds
                char_span = tokenized_output.token_to_chars(sentence_id, token_id)

                # Now, we go over all spaCy-tokenised tokens
                spacy_token_id = 0
                for spacy_token in spacy_tokens:
                   # Check if our char_span falls within the bounds of this spacy token
                    if char_span.start >= spacy_token.idx and char_span.end <= spacy_token.idx + len(spacy_token.text):
                        # If it does, it means that this word piece is part of the word spacy tokenised
                        # Add its id to the correspondence dict
                        if not spacy_token_id in correspondence:
                            correspondence[spacy_token_id] = [ token_id ]
                        else:
                            correspondence[spacy_token_id].append(token_id)
                            
                        # We need to use token ids instead of the token text, since a specific token may occur more
                        # than once in a sentence (e.g. I really really do not like this)
                    
                    spacy_token_id += 1
                    
                # Edge case
                #print(char_span.end)
                if char_span.end == len(tokenizer_input_strings[sentence_id]):
                    break

                token_id += 1
            
            correspondence_list.append(correspondence)

            
        # Return both the vector ids (which we will need for inference) as well as the correspondence dict
        return { "tokenized_sentence": tokenized_output,
                 "correspondence": correspondence_list,
                 "spacy_tokens": spacy_tokens_list }