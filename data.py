import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import re
from tqdm import tqdm
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"

class BOWTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word_to_index = {PAD_TOKEN: 0}
        self.index_to_word = {0: PAD_TOKEN}

    def fit(self, data):
        word_freq = {}

        for text, _ in data:
            for word in text:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

        words_by_freq = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)
        words_by_freq = words_by_freq[:(self.vocab_size - 1)]

        for index, word in enumerate(words_by_freq):
            self.word_to_index[word] = index + 1
            self.index_to_word[index + 1] = word

    def transform(self, text):
        if isinstance(text, str):
            text = text.split()
        return [self.word_to_index[word] for word in text if word in self.word_to_index]

class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer=None, vocab_size=10000, min_seq_length=0, max_seq_length=400, dataset_length=None, pad=True, quiet=False):
        self.quiet = quiet
        self.raw_data = data
        self.vocab_size = vocab_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.dataset_length = dataset_length

        if dataset_length and dataset_length < len(self.raw_data):
            self.raw_data = self.raw_data[:dataset_length]
        
        nltk.download('stopwords', quiet=True)
        self.tokenizer = tokenizer
        self.data = self._preprocess_data()
        self.data = self._tokenize_data()
        if pad:
            self.data = self._pad_sequences()
        self.data = self._convert_to_tensors()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def _preprocess_data(self):
        preprocessed_data = []
        data_iterator = self.raw_data
        if not self.quiet:
            data_iterator = tqdm(self.raw_data, desc="Preprocessing Text", ascii=' >=')
        
        # TODO: Preprocess the data
        for text, label in data_iterator:

            # Lowercase the text
            text = text.lower()

            # Remove html tags - anything between < and > i.e. <html>, </html>, etc.
            # You can use re.sub for a one-liner but doesn't matter
            text = re.sub(r'<.*?>', '', text) 

            # Remove punctuation - anything in string.punctuation
            text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)

            # Remove any words that are stopwords
            # Stopwords are common words that don't carry much information
            # Here's the list from nltk: https://gist.github.com/sebleier/554280
            stop_words = set(stopwords.words('english'))
            text = " ".join([word for word in text.split() if word not in stop_words])

            # Stemming is the process of reducing words to their root form
            # Here's an explanation of stemming: https://www.ibm.com/think/topics/stemming#:~:text=Stemming%20is%20a%20text%20preprocessing,a%20“lemma”%20in%20linguistics
            # We don't want "dancing" and "danced" to be different words in our vocabulary
            # We use PorterStemmer from nltk: https://www.nltk.org/api/nltk.stem.porter.html#nltk.stem.porter.PorterStemmer
            stemmer = PorterStemmer()
            text = " ".join([stemmer.stem(word) for word in text.split()])

            # Split the text by space
            # Hint: use the split() method
            text = text.split()

            preprocessed_data.append((text, label)) 
        if not self.quiet:
            data_iterator.close()
        return preprocessed_data
    
    def _tokenize_data(self):
        if not self.tokenizer:
            # initialize the tokenizer
            self.tokenizer = BOWTokenizer(vocab_size=self.vocab_size)

            # fit the tokenizer
            self.tokenizer.fit(self.data)

        # transform each list of strings into indices using the tokenizer
        transformed_data = [(self.tokenizer.transform(text), label) for text, label in self.data]
        
        return transformed_data
    
    # TODO: Pad the sequences so all the sequence in a batch are of the same length
    def _pad_sequences(self):

        # Every sequence in this list will be max_seq_length long
        padded_data = []        
            
        for indices, label in self.data:

            # Skip sequences that are shorter than min_seq_length
            if len(indices) < self.min_seq_length:
                continue
                
            # If the sequence of indicies is longer than max_seq_length, truncate it
            if len(indices) > self.max_seq_length:
                indices = indices[:self.max_seq_length]
            else: 
            # Otherwise, pad the indicies by appending PAD_TOKEN to the beginning of the indices sequence
                pad = self.max_seq_length - len(indices)
                indices = [0] * pad + indices
            
            
            padded_data.append((indices, label))

        return padded_data

    def _convert_to_tensors(self):
        return [(torch.tensor(indices), label) for indices, label in self.data]

class IMDBDataset(TextClassificationDataset):
    def __init__(self, split='train', dataset_length=-1, tokenizer=None, min_seq_length=0, max_seq_length=200):
        dataset = tfds.load('imdb_reviews', split=split, batch_size=-1, as_supervised=True)
        reviews, labels = tfds.as_numpy(dataset)
        data = list(zip([review.decode('utf-8') for review in reviews], labels))
        super().__init__(
            data,
            tokenizer=tokenizer,
            min_seq_length=min_seq_length,
            max_seq_length=max_seq_length,
            dataset_length=None if dataset_length == -1 else dataset_length
        )

def load_mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Loading MNIST dataset...")
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset


    