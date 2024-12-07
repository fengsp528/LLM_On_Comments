
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CommentScoreDataset(Dataset):
    def __init__(self, comments, scores, tokenizer, max_length=128):
        self.comments = comments
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = self.comments[idx]
        score = self.scores[idx]

        encoding = self.tokenizer(
            comment,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'score': torch.tensor(score, dtype=torch.float)
        }


class CommentScorePredictor(nn.Module):
    def __init__(self, bert_model="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        score = self.regressor(pooled_output)
        return score.squeeze()


# Example training data
example_comments = [
    "This is absolutely amazing! Life-changing content!",  # Score: 10
    "Great work, very helpful and informative",  # Score: 7
    "Nice post",  # Score: 3
    "okay",  # Score: 0
    "This is not very helpful",  # Score: -3
    "Terrible content, waste of time!",  # Score: -7
    "You're an idiot, this is garbage!!!"  # Score: -10
]

example_scores = [10.0, 7.0, 3.0, 0.0, -3.0, -7.0, -10.0]


def train_model(model, train_loader, num_epochs=5, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)

            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, scores)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CommentScoreDataset(example_comments, example_scores, tokenizer)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = CommentScorePredictor()
    train_model(model, train_loader)

    # Example prediction
    model.eval()
    test_comments = [
        "You are so bad!",
        "This is OK.",
        "You are the best in the world!"
    ]
    for test_comment in test_comments:
        inputs = tokenizer(test_comment, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            score = model(inputs['input_ids'], inputs['attention_mask'])
            print(f"Predicted score for '{test_comment}': {score.item():.2f}")


if __name__ == "__main__":
    main()
