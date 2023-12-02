import argparse
import torch
import torch.nn as nn
from transformers import DistilBertModel,  DistilBertTokenizer
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description='Make a prediction with a trained model.')
    parser.add_argument("--tweet", help="The tweet to classify.")
    parser.add_argument("--model_path", default="./my-model.pth", help="The path to the saved model.")
    args = parser.parse_args()

    if args.tweet is None:
        parser.print_help()
        return


    def softmax_focal_loss_with_regularization(inputs, targets, model, alpha=None, gamma=2, reduction="none",
                                               reg_type: str = "l2", reg_weight: float = 1e-3):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)
        loss = ce_loss * ((1 - p) ** gamma)

        # Alpha weighting for different classes
        if alpha is not None:
            batch_size, num_classes = inputs.size()
            class_mask = torch.zeros((batch_size, num_classes), device=inputs.device)
            class_mask.scatter_(1, targets.unsqueeze(1), 1.)
            alpha_t = torch.sum(alpha.to(inputs.device) * class_mask.to(inputs.device), dim=1)
            loss = alpha_t * loss

        # L1/L2 regularization
        reg_loss = 0
        for param in model.parameters():
            if reg_type == "l1":
                reg_loss += torch.sum(torch.abs(param))
            else:
                reg_loss += torch.sum(param ** 2)
        loss = loss + reg_weight * reg_loss

        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \\n Supported reduction modes: 'none', 'mean', 'sum'")

        return loss

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
            loss = softmax_focal_loss_with_regularization(logits.view(-1, 3), labels.view(-1), self)

            return (loss.mean(), logits)


    model = CustomClassificationModel()
    model.load_state_dict(torch.load(args.model_path))

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