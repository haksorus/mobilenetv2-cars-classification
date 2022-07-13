import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import LabelEncoder



def incorrect_samples(model, loader, device):
    """Get incorrectly classified paths and images"""
    incorrects, inc_paths = [], []
    cnt = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
          x, y = batch
          for image, label in zip(x, y):
            image = image.unsqueeze(0).to(device)
            outp = model(image).cpu()
            pred = outp.argmax(-1)
            if pred != label:
              incorrects.append((image.cpu(), label, pred))
              inc_paths.append(loader.dataset.samples[cnt])
            cnt += 1

    return inc_paths, incorrects


def class_score(model, loader, dataset, device):
    """Get accuracy-score for each class"""
    label_encoder = LabelEncoder().fit(dataset.classes)
    keys = label_encoder.classes_
    classes_predict = {key: [] for key in keys}
    classes_correct = {key: [] for key in keys}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            inputs, labels = batch
            inputs = inputs.to(device)
            outp = model(inputs).cpu()
            preds = outp.argmax(-1)
            for pred, label in zip(preds, labels):
                key = label_encoder.inverse_transform([label]).item()
                classes_predict[key].append(pred)
                classes_correct[key].append(label)

    data = []
    for key in keys:
        data.append((key, len(classes_correct[key]), round(accuracy_score(classes_predict[key], classes_correct[key]), ndigits=5)))

    return pd.DataFrame(columns=['Class', 'N', 'Accuracy'], data=data).sort_values(by='Accuracy')


def predict(model, loader,  device):
    "Get prediction"
    model.eval()
    with torch.no_grad():
        logits = []
    
        for inputs in loader:
            inputs = inputs.to(device)
            outp = model(inputs).cpu()
            logits.append(outp)
            
    preds = torch.cat(logits).numpy()

    return preds