from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
import numpy as np

def train_and_predict(model_name, texts, labels, new_sentence):
    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 classes: binary classification

    # Tokenize and prepare input tensors
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    labels = torch.tensor(labels).unsqueeze(1)  # Add an extra dimension for labels

    # Create a DataLoader for batching and shuffling
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Define loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop
    epochs = 3
    epoch_losses = []  # List to store training losses for each epoch

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            input_ids, attention_mask, label = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        epoch_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")

    # Plotting training loss
    plt.plot(range(1, epochs + 1), epoch_losses, label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.show()

    # Validation Loss (hypothetical, you need a validation dataset for this)
    # validation_loss = compute_validation_loss(model, val_dataloader)
    # print(f"Validation Loss for {model_name}: {validation_loss}")

    # Prediction
    with torch.no_grad():
        inputs = tokenizer([new_sentence], return_tensors='pt', padding=True, truncation=True)
        logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()

    print(f"Predicted class for '{new_sentence}' using {model_name}: {predictions.item()}")
    print(f"Prediction Probabilities: {probabilities}")
    print("=" * 50)

# Example data
texts = ["This is a positive sentence.", "This is a negative sentence."]
labels = [1, 0]  # 1 for positive, 0 for negative
new_sentence = "This is another sentence."

# Train and predict using different BERT models
models_to_try = ['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased', 'bert-base-multilingual-cased']

# Plotting the training loss for each model
for model_name in models_to_try:
    print(f"Training and predicting with {model_name}")
    train_and_predict(model_name, texts, labels, new_sentence)

# Bar chart comparing prediction probabilities for the new sentence
predictions_probabilities = []
for model_name in models_to_try:
    with torch.no_grad():
        inputs = tokenizer([new_sentence], return_tensors='pt', padding=True, truncation=True)
        logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
        predictions_probabilities.append(probabilities)

model_labels = [f"{model}\n{np.argmax(probabilities)}" for model, probabilities in zip(models_to_try, predictions_probabilities)]
bar_width = 0.2
index = np.arange(len(models_to_try))

plt.bar(index, predictions_probabilities, width=bar_width, label=model_labels)
plt.xlabel('Models')
plt.ylabel('Prediction Probabilities')
plt.title('Comparison of Prediction Probabilities for the New Sentence')
plt.xticks(index + bar_width / 2, models_to_try)
plt.legend()
plt.show()
