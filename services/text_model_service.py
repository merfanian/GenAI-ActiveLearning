import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re

from utils.config import TRAINED_MODELS_DIR

_current_model_path = "trained_models/text_model.pth"

def build_vocab(texts, min_freq=2):
    words = [word for text in texts for word in re.findall(r'\w+', text.lower())]
    word_counts = Counter(words)
    vocab = {word for word, count in word_counts.items() if count >= min_freq}
    return {word: i + 2 for i, word in enumerate(vocab)}, {i + 2: word for i, word in enumerate(vocab)}

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, label_to_idx):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized_text = [self.vocab.get(word, 1) for word in re.findall(r'\w+', text.lower())] # 1 for <unk>
        return torch.tensor(tokenized_text), self.label_to_idx[label]

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # 0 for <pad>
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    return torch.tensor(label_list, dtype=torch.int64), pad_sequence(text_list, batch_first=True, padding_value=0), torch.tensor(lengths)

def train_model(texts: list[str], labels: list[str], existing_model_path: str = None, updated_model_path: str = None) -> str:
    logging.debug(f"train_model called with {len(texts)} texts, existing_model_path={existing_model_path}")
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(list(set(labels)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    vocab, idx_to_vocab = build_vocab(texts)
    dataset = TextDataset(texts, labels, vocab, label_to_idx)

    def collate_fn(batch):
        texts, labels = zip(*batch)
        text_lengths = torch.tensor([len(text) for text in texts])
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        return padded_texts, torch.tensor(labels), text_lengths

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = LSTMClassifier(len(vocab) + 2, 100, 256, len(unique_labels), 2, 0.5)
    if existing_model_path:
        checkpoint = torch.load(existing_model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    epochs = 5
    model.train()
    for epoch in range(epochs):
        for texts_batch, labels_batch, lengths_batch in loader:
            optimizer.zero_grad()
            predictions = model(texts_batch, lengths_batch).squeeze(1)
            loss = criterion(predictions, labels_batch)
            loss.backward()
            optimizer.step()

    model_file = TRAINED_MODELS_DIR / "text_model.pth" if not updated_model_path else TRAINED_MODELS_DIR / updated_model_path
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "label_to_idx": label_to_idx,
        "idx_to_label": {i: label for label, i in label_to_idx.items()}
    }, model_file)
    logging.debug(f"Saved trained text model to {model_file}")
    return str(model_file)

def predict(model_path: str, texts: list[str]) -> list[dict]:
    logging.debug(f"predict called with model_path={model_path}, {len(texts)} texts")
    checkpoint = torch.load(model_path, map_location="cpu")
    vocab = checkpoint['vocab']
    idx_to_label = checkpoint['idx_to_label']
    
    model = LSTMClassifier(len(vocab) + 2, 100, 256, len(idx_to_label), 2, 0.5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []
    with torch.no_grad():
        for text in texts:
            tokenized = [vocab.get(word, 1) for word in re.findall(r'\w+', text.lower())]
            tensor = torch.LongTensor(tokenized).unsqueeze(0)
            length = torch.tensor([len(tokenized)])
            prediction = model(tensor, length)
            probs = torch.softmax(prediction, dim=1).squeeze().tolist()
            pred_idx = torch.argmax(prediction, dim=1).item()
            pred_label = idx_to_label[pred_idx]
            prob_dict = {idx_to_label[i]: prob for i, prob in enumerate(probs)}
            results.append({"predicted_label": pred_label, "probabilities": prob_dict})
    return results

def set_current_model_path(model_path: str):
    global _current_model_path
    _current_model_path = model_path

def get_current_model_path() -> str:
    return _current_model_path
