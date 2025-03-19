import argparse
import torch
from psthree.data import BOWTokenizer, TextClassificationDataset
from torch.utils.data import DataLoader

def test_reviews(model_path):

    tokenizer = BOWTokenizer(vocab_size=10000)
    
    model = torch.load(model_path, weights_only=False)
    tokenizer = model.tokenizer
    
    model.eval()

    print("Enter a review (type 'quit' to exit):")
    
    while True:
        review = input("\nReview: ").strip()
        
        if review.lower() == 'quit':
            break
        dataset = TextClassificationDataset(tokenizer=tokenizer, data=[(review, 0)], pad=False, quiet=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            text, _ = next(iter(dataloader))
            if text.shape[1] == 0:
                print("Review is empty")
                continue
            outputs = model(text)
            outputs = outputs[:, -1, :]
            prediction = torch.argmax(outputs, dim=-1)
            confidence_score = outputs[0][prediction].item() * 100
            sentiment = "Positive" if prediction == 1 else "Negative"
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence_score:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()

    test_reviews(args.model_path)
