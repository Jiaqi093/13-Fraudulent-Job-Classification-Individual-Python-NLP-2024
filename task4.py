import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Define the dataset class
class JobPostingDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_len=512):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, index):
        description = str(self.descriptions.iloc[index])  # Use iloc to ensure proper row indexing
        label = self.labels.iloc[index]  # Use iloc here as well for correct label retrieval

        encoding = self.tokenizer(
            description,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            'description_text': description,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv(r'data\fake_job_postings.csv')

# User can choose the task (binary classification using 'fraudulent' or multi-class using 'required_education')
#task = 'fraudulent'
task = 'required_education'

# Preprocess the dataset (get descriptions and labels)
descriptions = data['description']

# Handle different tasks
if task == 'fraudulent':
    labels = data['fraudulent']  # Binary classification with 0 and 1 labels
    num_labels = 2
else:
    # Multi-class classification using 'required_education'
    data = data[data['fraudulent'] == 0]
    data = data[data['required_education'].isin(["Master's Degree", "Bachelor's Degree", "High School or equivalent"])]
    data = data.reset_index(drop=True)  # Reset the index after filtering to avoid misalignment
    descriptions = data['description']
    label_mapping = {
        "High School or equivalent": 0,
        "Bachelor's Degree": 1,
        "Master's Degree": 2
    }
    labels = data['required_education'].map(label_mapping)
    num_labels = 3

# Initialize tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Split the dataset into 70% train, 10% validation, and 20% test
train_size = 0.7
val_size = 0.1
test_size = 0.2

train_len = int(len(descriptions) * train_size)
val_len = int(len(descriptions) * val_size)
test_len = len(descriptions) - train_len - val_len

# Train, validation, and test splits
train_descriptions = descriptions[:train_len]
train_labels = labels[:train_len]

val_descriptions = descriptions[train_len:train_len+val_len]
val_labels = labels[train_len:train_len+val_len]

test_descriptions = descriptions[train_len+val_len:]
test_labels = labels[train_len+val_len:]

# Create dataset objects for train, validation, and test
train_dataset = JobPostingDataset(train_descriptions, train_labels, tokenizer)
val_dataset = JobPostingDataset(val_descriptions, val_labels, tokenizer)
test_dataset = JobPostingDataset(test_descriptions, test_labels, tokenizer)

# Initialize model based on the number of labels
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)

# Finetuning function
def finetune_model(model, dataset_train, dataset_val=None, eval_fn=None, batch_size=8, n_epochs=2, learning_rate=1e-5):
    model.train(True)

    # Create dataloaders
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size) if dataset_val else None

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        total_loss = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['label'].to(device)
            )

            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Average training loss: {total_loss / len(train_dataloader)}")

        # Evaluate on validation set
        if val_dataloader and eval_fn:
            accuracy, precision, recall, f1 = eval_fn(model, val_dataloader)
            print(f'Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    return model

# Evaluation function (returns accuracy, precision, recall, and F1 score)
def evaluate_model(model, dataloader):
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            preds = torch.argmax(outputs.logits, dim=1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(batch['label'].cpu().numpy())

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds, average='weighted')
    recall = recall_score(total_labels, total_preds, average='weighted')
    f1 = f1_score(total_labels, total_preds, average='weighted')

    return accuracy, precision, recall, f1

# Load original DistilBERT and evaluate without fine-tuning
print("Evaluating original (pre-trained) DistilBERT on test set:")
original_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
original_model.to(device)

# Evaluate original DistilBERT on the test dataset
accuracy, precision, recall, f1 = evaluate_model(original_model, DataLoader(test_dataset, batch_size=8))
print(f"Original DistilBERT - Test Accuracy: {accuracy:.4f}")
print(f"Original DistilBERT - Test Precision: {precision:.4f}")
print(f"Original DistilBERT - Test Recall: {recall:.4f}")
print(f"Original DistilBERT - Test F1 Score: {f1:.4f}")

# Finetune DistilBERT and evaluate after fine-tuning
print("\nFine-tuning DistilBERT on training data:")
fine_tuned_model = finetune_model(model, train_dataset, val_dataset, eval_fn=evaluate_model, n_epochs=3)

# Evaluate fine-tuned DistilBERT on the test dataset
accuracy, precision, recall, f1 = evaluate_model(fine_tuned_model, DataLoader(test_dataset, batch_size=8))
print(f"Fine-tuned DistilBERT - Test Accuracy: {accuracy:.4f}")
print(f"Fine-tuned DistilBERT - Test Precision: {precision:.4f}")
print(f"Fine-tuned DistilBERT - Test Recall: {recall:.4f}")
print(f"Fine-tuned DistilBERT - Test F1 Score: {f1:.4f}")
