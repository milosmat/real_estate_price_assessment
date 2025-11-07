import os
import time
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb

# Definisanje funkcije za učitavanje checkpoint-a (samo model)
def load_checkpoint(model, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        start_time = time.time()
        checkpoint = torch.load(filename)
        print(f"Checkpoint loaded in {time.time() - start_time} seconds")
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0

# Učitavanje podataka
data = pd.read_csv('filtered_property_data.csv')
original_ids = data['id']
data = data.drop_duplicates()
data = data.drop(columns=['id'])
data = data.dropna(subset=['kvadratura', 'cena'])

# Pretprocesiranje podataka
label_encoders = {}
for column in ['grad', 'opstina', 'kvart', 'broj_soba', 'spratnost', 'stanje', 'grejanje', 'lift', 'podrum']:
    label_encoders[column] = LabelEncoder()
    data[column] = data[column].fillna('missing')
    data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Odabir relevantnih kolona
best_features = ['kvadratura', 'grad', 'opstina', 'kvart', 'broj_soba', 'spratnost', 'grejanje', 'lift', 'podrum']
target = data['cena']

# Skaliranje kvadrature i cene
feature_scaler = StandardScaler()
data[best_features] = feature_scaler.fit_transform(data[best_features])

price_scaler = StandardScaler()
data[['cena']] = price_scaler.fit_transform(data[['cena']])

# Učitavanje sačuvanih najboljih parametara za XGBoost model
best_params = joblib.load('best_params_xgboost_random_search.pkl')

# Definisanje transformacija za slike
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Definisanje dataset klase za slike
class PropertyImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.prices = []

        for folder in ['kupatilo', 'kuhinja', 'sobe', 'terasa']:
            folder_path = os.path.join(image_dir, folder)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_id = img_name.split('_')[0]
                    img_path = os.path.join(folder_path, img_name)
                    self.image_paths.append(img_path)
                    price = self.data[self.data['id'] == img_id]['cena']
                    if not price.empty:
                        self.prices.append(price.values[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        price = self.prices[idx]
        price = torch.tensor(price, dtype=torch.float)
        
        return image, price

# Kreiranje dataset-a i data loader-a
image_dir = 'property_images'
csv_file = 'filtered_property_data.csv'
dataset = PropertyImageDataset(csv_file=csv_file, image_dir=image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Učitavanje pretreniranog ResNet modela
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in resnet.parameters():
    param.requires_grad = False

# Modifikacija poslednjeg sloja za ekstrakciju karakteristika
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 1)

# Inicijalizacija optimizatora za ResNet model
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# Učitavanje checkpoint-a za ResNet model ako postoji
start_epoch = load_checkpoint(resnet, filename="checkpoint.pth.tar")

# Kreiranje kombinovanog modela
class CombinedModel(BaseEstimator, RegressorMixin):
    def __init__(self, xgb_model, resnet_model, dataloader, scaler):
        self.xgb_model = xgb_model
        self.resnet_model = resnet_model
        self.dataloader = dataloader
        self.scaler = scaler

    def fit(self, X, y):
        self.xgb_model.fit(X, y)

    def predict(self, X):
        xgb_preds = self.xgb_model.predict(X)
        self.resnet_model.eval()
        resnet_preds = []

        with torch.no_grad():
            for images, _ in self.dataloader:
                outputs = self.resnet_model(images).squeeze().numpy()
                resnet_preds.extend(outputs)

        resnet_preds = np.array(resnet_preds)
        resnet_preds = price_scaler.inverse_transform(resnet_preds.reshape(-1, 1)).flatten()
        combined_preds = (xgb_preds + resnet_preds[:len(xgb_preds)]) / 2
        return combined_preds

# Kreiranje modela sa najboljim parametrima
xgb_model = xgb.XGBRegressor(**best_params, random_state=42)

# Kreiranje kombinovanog modela
combined_model = CombinedModel(xgb_model, resnet, dataloader, price_scaler)

# Treniranje kombinovanog modela
X_train, X_test, y_train, y_test = train_test_split(data[best_features], target, test_size=0.2, random_state=42)
combined_model.fit(X_train, y_train)

# Predikcije sa kombinovanim modelom
predictions_combined = combined_model.predict(X_test)

# Vizualizacija predikcija
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions_combined, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.xlabel('Actual Prices (EUR)')
plt.ylabel('Predicted Prices (EUR)')
plt.title('Actual vs Predicted Prices with Combined Model')
plt.show()
