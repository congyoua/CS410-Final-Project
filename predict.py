import argparse
import torch
import torch.nn as nn
from transformers import DistilBertModel,  DistilBertTokenizer

def main():
    parser = argparse.ArgumentParser(description='Make a prediction with a trained model.')
    parser.add_argument("--tweet", help="The tweet to classify.")
    parser.add_argument("--model_path", default="./my-model.pth", help="The path to the saved model.")
    args = parser.parse_args()

    if args.tweet is None:
        parser.print_help()
        return

    class CustomClassificationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.pre_classifier = nn.Linear(self.distilbert.config.dim, 256)
            self.dropout = nn.Dropout(0.2)
            self.classifier1 = nn.Linear(256, 64)
            self.classifier2 = nn.Linear(64, 3)

        def forward(self, input_ids, attention_mask=None, labels=None):
            distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = distilbert_output[0]
            out = hidden_state[:, 0]
            out = self.pre_classifier(out)
            out = nn.ReLU()(out)
            out = self.dropout(out)
            out = self.classifier1(out)
            out = nn.ReLU()(out)
            out = self.dropout(out)
            logits = self.classifier2(out)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            return (loss.mean(), logits)

    model = CustomClassificationModel()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    # Prepare the inputs
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(
        args.tweet,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    inputs.update({"labels":torch.tensor([0])})

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label_index = outputs[1].argmax().item()
    label_map = {i:v for i, v in enumerate(['hate_speech','offensive_language', 'neither'])}
    # Output the final class in string
    print("This tweet is", label_map[predicted_label_index])


if __name__ == "__main__":
    main()