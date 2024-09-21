import pandas as pd
import os

def convert_to_binary_sentiment(input_file, output_dir):
    print(f"Processing file: {input_file}")
    df = pd.read_csv(input_file)

    # Check if required columns exist
    required_columns = ['review_text', 'class_index']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")

    print("Converting ratings to binary sentiment...")
    df['sentiment'] = df['class_index'].apply(lambda x: 1 if x >= 4 else 0)

    # Create a new DataFrame with required columns
    new_df = df[['review_text', 'sentiment']]

    # Generate output file name
    input_filename = os.path.basename(input_file)
    output_filename = f"processed_data_{input_filename}"
    output_file = os.path.join(output_dir, output_filename)

    print(f"Saving output file: {output_file}")
    new_df.to_csv(output_file, index=False)

    # Print statistics
    total_reviews = len(new_df)
    positive_reviews = sum(new_df['sentiment'] == 1)
    negative_reviews = sum(new_df['sentiment'] == 0)

    print(f"Total reviews: {total_reviews}")
    print(f"Positive reviews (1): {positive_reviews} ({positive_reviews/total_reviews:.2%})")
    print(f"Negative reviews (0): {negative_reviews} ({negative_reviews/total_reviews:.2%})")

    return output_file

def process_files(train_file, test_file):
    # Create output directory
    output_dir = os.path.join(os.path.dirname(train_file), "processed_data")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # Process train file
    print("\nProcessing train file...")
    train_output = convert_to_binary_sentiment(train_file, output_dir)

    # Process test file
    print("\nProcessing test file...")
    test_output = convert_to_binary_sentiment(test_file, output_dir)

    return train_output, test_output

if __name__ == "__main__":
    # Get input file paths from user
    train_file = input("Enter the path to your train CSV file: ").strip()
    test_file = input("Enter the path to your test CSV file: ").strip()

    # Validate input files
    for file in [train_file, test_file]:
        if not os.path.isfile(file) or not file.lower().endswith('.csv'):
            print(f"Error: Invalid file path or not a CSV file: {file}")
            exit(1)

    try:
        train_output, test_output = process_files(train_file, test_file)
        print("\nProcessing complete!")
        print(f"Processed train file saved as: {train_output}")
        print(f"Processed test file saved as: {test_output}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        exit(1)
