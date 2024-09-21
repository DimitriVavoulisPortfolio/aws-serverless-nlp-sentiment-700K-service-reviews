# MODEL.md - Sentiment Analysis Model for Service Reviews Documentation

## Model Architecture

The sentiment analysis model is based on DistilBERT, a distilled version of BERT that retains 97% of BERT's language understanding capabilities while being 40% smaller and 60% faster. 

- Base model: DistilBERT (distilbert-base-uncased)
- Task: Sequence Classification (Binary Sentiment Analysis)
- Output: 2 classes (Positive/Negative sentiment)

## Training Process

### Dataset
- Total samples: 700K Yelp service reviews
- Features: 'review_text' (preprocessed review text), 'sentiment' (0 for negative, 1 for positive, after processing from 1-5 scale)

### Training Configuration
- Framework: PyTorch
- Batch size: 32
- Maximum epochs: 10
- Learning rate strategy:
  - Warm-up steps: 500
  - Schedule: Linear warmup followed by linear decay
  - Initial learning rate: Not explicitly specified, uses the default from the Hugging Face Trainer
- Weight decay: 0.01
- Mixed precision training: Enabled (fp16)

### Custom Training Features
- Minimum epochs: 5
- Early stopping patience: 2
- Evaluation strategy: Per epoch
- Save strategy: Best model at the end of training

## Model Conversion

After training in PyTorch, the model was converted to TensorFlow format for deployment flexibility.

Conversion process:
1. Load trained PyTorch model
2. Convert to TensorFlow using Hugging Face's `from_pt=True` option
3. Save TensorFlow model and tokenizer

## Quantization

No explicit quantization was performed after the initial training and conversion. The model uses the default precision of the DistilBERT architecture.

## Performance Metrics

Based on the test results shown in the image:

- Accuracy: 0.8933 (89.33%)
- Precision: 0.8536 (85.36%)
- Recall: 0.8851 (88.51%)
- F1 Score: 0.8691 (86.91%)

## Additional Notes

- The model was trained on a CUDA-enabled device, leveraging GPU acceleration for faster training.
- The learning rate strategy employed (linear warmup followed by linear decay) is a common practice in fine-tuning transformer models. It helps stabilize training in the early stages and allows for better convergence.
- The high performance metrics suggest that the model generalizes well to unseen data, though it's important to continually monitor for potential overfitting in real-world applications.
- Future work may include experimenting with different learning rate schedules, explicitly setting initial learning rates, exploring other model architectures, fine-tuning hyperparameters, or investigating quantization techniques to optimize for deployment in resource-constrained environments like AWS Lambda.

