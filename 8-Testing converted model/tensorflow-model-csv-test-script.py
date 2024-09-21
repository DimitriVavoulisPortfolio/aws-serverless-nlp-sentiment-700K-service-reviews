import pandas as pd
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("GPU is available. Using GPU.")
    device = "/GPU:0"
else:
    print("GPU is not available. Using CPU.")
    device = "/CPU:0"

# Set the model path and CSV file path
model_path = r"C:\Users\dimit\Downloads\AWS Serverless Sentiment Analysis for yelp service reviews\7-tensorflow_model"
csv_file_path = r"C:\Users\dimit\Downloads\AWS Serverless Sentiment Analysis for yelp service reviews\2-CSV files\processed_data\processed_data_code_test.csv"

# Load the model and tokenizer
with tf.device(device):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Maximum sequence length and batch size
MAX_LENGTH = 128
BATCH_SIZE = 32

def preprocess_text(texts):
    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    return encoded['input_ids'], encoded['attention_mask']

def predict_sentiment_batch(texts):
    input_ids, attention_mask = preprocess_text(texts)
    with tf.device(device):
        input_ids = tf.convert_to_tensor(input_ids)
        attention_mask = tf.convert_to_tensor(attention_mask)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1)
        predicted_classes = tf.argmax(probabilities, axis=-1).numpy()
        confidence_scores = tf.reduce_max(probabilities, axis=-1).numpy()
    return predicted_classes, confidence_scores

def process_in_batches(df):
    all_predictions = []
    all_confidences = []
    
    for i in range(0, len(df), BATCH_SIZE):
        batch = df['review_text'][i:i+BATCH_SIZE].tolist()
        predictions, confidences = predict_sentiment_batch(batch)
        all_predictions.extend(predictions)
        all_confidences.extend(confidences)
        
        if i % 1000 == 0:
            print(f"Processed {i}/{len(df)} reviews...")
    
    return all_predictions, all_confidences

def main():
    print("Loading CSV file...")
    df = pd.read_csv(csv_file_path)
    
    print("Processing and making predictions...")
    predicted_classes, confidence_scores = process_in_batches(df)
    
    true_labels = df['sentiment'].tolist()
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_classes)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_classes, target_names=['Negative', 'Positive']))
    
    # Add predictions to the DataFrame
    df['predicted_sentiment'] = predicted_classes
    df['confidence'] = confidence_scores
    
    # Save results to a new CSV file
    output_file = 'sentiment_analysis_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print some example predictions
    print("\nExample Predictions:")
    for i in range(min(10, len(df))):
        print(f"Text: {df['review_text'].iloc[i]}")
        print(f"True Sentiment: {df['sentiment'].iloc[i]}")
        print(f"Predicted Sentiment: {predicted_classes[i]}")
        print(f"Confidence: {confidence_scores[i]:.4f}")
        print()

if __name__ == "__main__":
    main()

# This file contains code that uses the "Yelp Reviews for SA fine-grained 5 classes CSV" dataset,
# which is licensed under the Apache License 2.0. 
# For more information, see: https://www.kaggle.com/datasets/yacharki/yelp-reviews-for-sa-finegrained-5-classes-csv
