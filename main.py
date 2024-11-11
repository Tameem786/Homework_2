import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from collections import Counter

# Set the device to GPU if available, otherwise use CPU.
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

# Expanded training data pairs (English and Spanish sentences)
data = [
    # Basic greetings
    ["hello", "hola"],
    ["hi", "hola"],
    ["hello friend", "hola amigo"],
    ["good morning", "buenos días"],
    ["good afternoon", "buenas tardes"],
    ["good evening", "buenas tardes"],
    ["good night", "buenas noches"],

    # Common phrases
    ["how are you", "cómo estás"],
    ["how are you doing", "cómo te va"],
    ["i am fine", "estoy bien"],
    ["thank you", "gracias"],
    ["thank you very much", "muchas gracias"],
    ["you are welcome", "de nada"],
    ["please", "por favor"],

    # Basic responses
    ["yes", "sí"],
    ["no", "no"],
    ["maybe", "quizás"],
    ["of course", "por supuesto"],
    ["i agree", "estoy de acuerdo"],

    # Introductions
    ["what is your name", "cuál es tu nombre"],
    ["my name is", "mi nombre es"],
    ["nice to meet you", "encantado de conocerte"],
    ["where are you from", "de dónde eres"],
    ["i am from", "soy de"],

    # Farewells
    ["goodbye", "adiós"],
    ["bye", "chao"],
    ["see you later", "hasta luego"],
    ["see you tomorrow", "hasta mañana"],
    ["see you soon", "hasta pronto"],

    # Common questions
    ["what time is it", "qué hora es"],
    ["how old are you", "cuántos años tienes"],
    ["where is the bathroom", "dónde está el baño"],
    ["how much is this", "cuánto cuesta esto"],

    # Basic feelings and states
    ["i am happy", "estoy feliz"],
    ["i am tired", "estoy cansado"],
    ["i am hungry", "tengo hambre"],
    ["i am thirsty", "tengo sed"],

    # Basic verbs
    ["i want", "quiero"],
    ["i need", "necesito"],
    ["i like", "me gusta"],
    ["i love", "amo"],

    # Time expressions
    ["today", "hoy"],
    ["tomorrow", "mañana"],
    ["yesterday", "ayer"],
    ["now", "ahora"],

    # Numbers and basic counting
    ["one", "uno"],
    ["two", "dos"],
    ["three", "tres"],
    ["four", "cuatro"],
    ["five", "cinco"]
]

# Test dataset with patterns similar to training data
test_data = [
    ["hello my friend", "hola mi amigo"],  # Similar to "hello friend"
    ["how are they doing", "cómo están"],  # Similar to "how are you doing"
    ["good evening friend", "buenas tardes amigo"],  # Similar to "good evening"
    ["thank you friend", "gracias amigo"],  # Similar to "thank you"
    ["see you next time", "hasta la próxima"],  # Similar to "see you later"
    ["good night friend", "buenas noches amigo"],  # Similar to "good night"
    ["please help me", "por favor ayúdame"],  # Similar to "please"
    ["yes of course", "sí por supuesto"],  # Similar to "yes" and "of course"
    ["no thank you", "no gracias"],  # Similar to "no" and "thank you"
    ["what is her name", "cuál es su nombre"],  # Similar to "what is your name"
    ["i am very happy", "estoy muy feliz"],  # Similar to "i am happy"
    ["i need help", "necesito ayuda"],  # Similar to "i need"
    ["see you tomorrow friend", "hasta mañana amigo"],  # Similar to "see you tomorrow"
    ["i want this", "quiero esto"],  # Similar to "i want"
    ["i like it", "me gusta esto"]  # Similar to "i like"
]

# Define a simple tokenizer to convert words to indices and vice versa
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, sentences):
        # Build a vocabulary from the provided sentences
        # Each word is mapped to a unique index, starting from 4 (0-3 reserved for special tokens)
        unique_words = set(word for sentence in sentences for word in sentence.split())
        self.word2idx = {word: idx + 4 for idx, word in enumerate(unique_words)}
        self.word2idx["<pad>"] = 0
        self.word2idx["<sos>"] = 1
        self.word2idx["<eos>"] = 2
        self.word2idx["<unk>"] = 3
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, sentence):
        # Convert a sentence into a list of token indices
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence.split()]

    def decode(self, indices):
        # Convert a list of token indices back into a sentence
        return " ".join([self.idx2word.get(idx, "<unk>") for idx in indices])


# Create tokenizers for English and Spanish languages
english_tokenizer = SimpleTokenizer()
spanish_tokenizer = SimpleTokenizer()

# Extract English and Spanish sentences from the data and build vocabularies
english_sentences = [pair[0] for pair in data]
spanish_sentences = [pair[1] for pair in data]

english_tokenizer.build_vocab(english_sentences)
spanish_tokenizer.build_vocab(spanish_sentences)


# Function to pad token sequences to a fixed length
def pad_sequence(seq, max_len, pad_idx=0):
    return seq + [pad_idx] * (max_len - len(seq))


# Preprocess the data by tokenizing and padding sentences
def preprocess_data(data, max_len):
    english_data = []
    spanish_data = []

    for eng, spa in data:
        # Encode sentences and add <sos> (1) and <eos> (2) tokens
        eng_tokens = [1] + english_tokenizer.encode(eng) + [2]  # <sos> ... <eos>
        spa_tokens = [1] + spanish_tokenizer.encode(spa) + [2]  # <sos> ... <eos>

        # Pad sequences to the maximum length
        eng_tokens = pad_sequence(eng_tokens, max_len)
        spa_tokens = pad_sequence(spa_tokens, max_len)

        english_data.append(eng_tokens)
        spanish_data.append(spa_tokens)

    # Convert the lists into PyTorch tensors
    return torch.tensor(english_data), torch.tensor(spanish_data)


# Define the maximum sequence length for input data
max_len = 10
english_tensor, spanish_tensor = preprocess_data(data, max_len)


# Define a dataset class for translation tasks
class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        # Return the number of data pairs
        return len(self.src_data)

    def __getitem__(self, idx):
        # Return source sentence, target input (without the last token), target output (without the first token)
        return self.src_data[idx], self.tgt_data[idx][:-1], self.tgt_data[idx][1:]


# Create a DataLoader to iterate through the dataset
dataset = TranslationDataset(english_tensor, spanish_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len):
        super(Transformer, self).__init__()
        # Initialize the encoder and decoder components
        self.encoder = TransformerEncoder(src_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len)
        # Output linear layer for vocab prediction
        self.output_layer = nn.Linear(embed_dim, tgt_vocab_size)
        self.device = device

    def forward(self, src, tgt):
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)
        return self.output_layer(decoder_output)


# Calculate 1-gram BLEU score
def calculate_1gram_bleu(candidate, reference):
    """
    Calculate 1-gram BLEU score

    Args:
        candidate (str): Model generated translation
        reference (str): Reference translation

    Returns:
        float: 1-gram BLEU score
    """
    # Split into words
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()

    # Count 1-grams for each sentence
    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)

    # Calculate matching 1-grams
    matches = sum(min(candidate_counts[word], reference_counts[word])
                  for word in candidate_counts)

    # Total number of generated words
    total_candidate_words = len(candidate_tokens)

    if total_candidate_words == 0:
        return 0.0

    # Calculate 1-gram precision
    bleu_1gram = matches / total_candidate_words

    return bleu_1gram


def evaluate_test_set(model, test_data, max_len=10):
    """
    Evaluate the model on the entire test dataset

    Args:
        model: Transformer model
        test_data: Test dataset
        max_len: Maximum sequence length

    Returns:
        float: Average 1-gram BLEU score
        list: List of (source, generated, reference, bleu_score) tuples
    """
    model.eval()
    total_bleu = 0.0
    results = []

    print("\nEvaluation on test set:")

    for src, tgt in test_data:
        translated = test(model, src, english_tokenizer, spanish_tokenizer, max_len)
        bleu_score = calculate_1gram_bleu(translated, tgt)
        total_bleu += bleu_score

        results.append((src, translated, tgt, bleu_score))

        print(f"Source: {src}")
        print(f"Generated: {translated}")
        print(f"Reference: {tgt}")
        print(f"1-gram BLEU: {bleu_score:.4f}\n")

    avg_bleu = total_bleu / len(test_data)
    print(f"Average 1-gram BLEU score: {avg_bleu:.4f}")

    return avg_bleu, results


# Set hyperparameters and instantiate the Transformer model
embed_dim = 512
num_heads = 8
num_layers = 2
hidden_dim = 2048

model = Transformer(
    src_vocab_size=english_tokenizer.vocab_size,
    tgt_vocab_size=spanish_tokenizer.vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    hidden_dim=hidden_dim,
    max_len=max_len
).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define model save path
MODEL_PATH = "transformer_translation_model.pth"


def train(model, dataloader, criterion, optimizer, num_epochs=10):
    """
    Model training function with evaluation on full test set at each epoch
    """
    best_bleu = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for src, tgt_input, tgt_output in dataloader:
            optimizer.zero_grad()

            # Move tensors to device
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            # Forward pass and loss calculation
            outputs = model(src, tgt_input)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print(f'Average Loss: {avg_loss:.4f}')

        # Evaluate on full test set
        print("\nTesting on all test samples:")
        avg_bleu, _ = evaluate_test_set(model, test_data)

        # Save model if it achieves better BLEU score
        if avg_bleu > best_bleu:
            best_bleu = avg_bleu
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'New best model saved with BLEU score: {best_bleu:.4f}')

        print('-' * 50)


# Convert input sentence to tensor for model
def tokenize_input(sentence, tokenizer, max_len):
    tokens = tokenizer.encode(sentence)
    tokens = [1] + tokens + [2]  # Add <sos> and <eos> tokens
    tokens = pad_sequence(tokens, max_len)
    return torch.tensor(tokens).unsqueeze(0)  # Add batch dimension


# Test function for translating sentences
def test(model, input_sentence, english_tokenizer, spanish_tokenizer, max_len=10):
    model.eval()
    src_tensor = tokenize_input(input_sentence, english_tokenizer, max_len).to(device)

    with torch.no_grad():
        encoder_output = model.encoder(src_tensor)
        tgt_tokens = torch.tensor([[1]]).to(device)  # Start with <sos> token

        translated_tokens = []
        for _ in range(max_len):
            with torch.no_grad():
                output = model.decoder(tgt_tokens, encoder_output)
                output = model.output_layer(output)
            next_token = output[:, -1, :].argmax(1).item()
            if next_token == 2:  # Stop if <eos> token is generated
                break
            translated_tokens.append(next_token)
            tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token]]).to(device)], dim=1)

        translated_sentence = spanish_tokenizer.decode(translated_tokens)
        return translated_sentence


# Main execution
if __name__ == "__main__":
    # Train model
    train(model, dataloader, criterion, optimizer, num_epochs=10)

    # Load best performing model
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    # Final evaluation
    print("\nFinal evaluation on test set:")
    final_bleu, final_results = evaluate_test_set(model, test_data)

    # Print summary of best and worst translations
    print("\nBest and Worst Translations:")
    sorted_results = sorted(final_results, key=lambda x: x[3], reverse=True)

    print("\nTop 3 Best Translations:")
    for src, gen, ref, bleu in sorted_results[:3]:
        print(f"Source: {src}")
        print(f"Generated: {gen}")
        print(f"Reference: {ref}")
        print(f"BLEU: {bleu:.4f}\n")

    print("\nTop 3 Worst Translations:")
    for src, gen, ref, bleu in sorted_results[-3:]:
        print(f"Source: {src}")
        print(f"Generated: {gen}")
        print(f"Reference: {ref}")
        print(f"BLEU: {bleu:.4f}\n")
