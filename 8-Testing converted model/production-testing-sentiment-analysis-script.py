from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import pandas as pd

# Set the model path and input file path
model_path = r"C:\Users\dimit\Downloads\AWS Serverless Sentiment Analysis for yelp service reviews\7-tensorflow_model"
input_file_path = r"C:\Users\dimit\Downloads\AWS Serverless Sentiment Analysis for yelp service reviews\2-CSV files\processed_data\processed_data_production_test.csv"

# Load the model and tokenizer
print("Loading model...")
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
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_classes = tf.argmax(probabilities, axis=-1).numpy()  # Now 0 or 1 directly
    confidence_scores = tf.reduce_max(probabilities, axis=-1).numpy()
    return predicted_classes, confidence_scores

def process_reviews(reviews):
    all_predictions = []
    all_confidences = []
    
    for i in range(0, len(reviews), BATCH_SIZE):
        batch = reviews[i:i+BATCH_SIZE]
        predictions, confidences = predict_sentiment_batch(batch)
        all_predictions.extend(predictions)
        all_confidences.extend(confidences)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(reviews)} reviews...")
    
    return all_predictions, all_confidences

def main():
    print("Reading input file...")
    df = pd.read_csv(input_file_path)
    
    # Ensure the column name is 'review_text'
    if 'review_text' not in df.columns:
        raise ValueError("The CSV file must contain a 'review_text' column")
    
    reviews = df['review_text'].tolist()
    
    print(f"Processing {len(reviews)} reviews...")
    predicted_classes, confidence_scores = process_reviews(reviews)
    
    # Calculate statistics
    total_reviews = len(reviews)
    positive_reviews = sum(1 for sentiment in predicted_classes if sentiment == 1)
    negative_reviews = total_reviews - positive_reviews
    avg_confidence = sum(confidence_scores) / total_reviews
    
    # Print report
    print("\n--- Sentiment Analysis Report ---")
    print(f"Total Reviews Analyzed: {total_reviews}")
    print(f"Positive Reviews: {positive_reviews} ({positive_reviews/total_reviews*100:.2f}%)")
    print(f"Negative Reviews: {negative_reviews} ({negative_reviews/total_reviews*100:.2f}%)")
    print(f"Average Confidence: {avg_confidence:.4f}")
    
    # Print some example predictions
    print("\nExample Predictions:")
    for i in range(min(5, len(reviews))):
        print(f"Review: {reviews[i][:50]}..." if len(reviews[i]) > 50 else reviews[i])
        print(f"Predicted Sentiment: {'Positive' if predicted_classes[i] == 1 else 'Negative'}")
        print(f"Confidence: {confidence_scores[i]:.4f}")
        print()

    # Add predictions to the DataFrame
    df['predicted_sentiment'] = predicted_classes
    df['confidence'] = confidence_scores
    
    # Save results to a new CSV file
    output_file = 'sentiment_analysis_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()