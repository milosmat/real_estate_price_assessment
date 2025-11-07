import os
import torch
from torchvision import models, transforms
from PIL import Image
import csv
from pathlib import Path

# Proširena mapa za kategorije koje odgovaraju tvojim potrebama
LABELS_MAP = {
    '/b/bathroom': 'kupatilo',
    '/s/shower': 'kupatilo',
    '/s/sauna': 'kupatilo',
    '/u/utility_room': 'kupatilo', # Dodato u mapu kupatilo
    '/l/laundromat': 'kupatilo', # Dodato u mapu kupatilo
    '/b/bedroom': 'sobe',
    '/b/bedchamber': 'sobe',
    '/c/classroom': 'sobe',
    '/c/conference_room': 'sobe',
    '/d/dorm_room': 'sobe',
    '/h/home_office': 'sobe',
    '/h/hospital_room': 'sobe',
    '/h/hotel_room': 'sobe',
    '/l/living_room': 'sobe',
    '/t/television_room': 'sobe',
    '/w/waiting_room': 'sobe',
    '/t/throne_room': 'sobe',
    '/m/music_studio': 'sobe',
    '/k/kitchen': 'kuhinja',
    '/r/restaurant_kitchen': 'kuhinja',
    '/b/balcony/exterior': 'terasa',
    '/b/balcony/interior': 'terasa',
    '/p/patio': 'terasa',
    '/r/restaurant_patio': 'terasa',
    '/r/roof_garden': 'terasa',
    '/g/gazebo/exterior': 'terasa'
}

# Učitajte kategorije iz datoteke
def load_labels(file_path='klasifikacija slika/categories_places365.txt'):
    labels = []
    with open(file_path) as f:
        for line in f:
            labels.append(line.strip().split(' ')[0])
    return labels

# Učitaj pretrenirani Places365 model
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 365)  # Zamenite poslednji sloj
    checkpoint = torch.load('klasifikacija slika/resnet18_places365.pth.tar', map_location='cpu')
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Definiši transformacije za ulazne slike
transform = transforms.Compose([
    transforms.Resize(256),  # Smanji rezoluciju na manju vrednost
    transforms.CenterCrop(224),  # Centrirano seče sliku
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Funkcija za klasifikaciju slike
def classify_image(model, image_path, labels):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    top5_labels = [(labels[catid], prob.item()) for prob, catid in zip(top5_prob, top5_catid)]
    
    print(f"Top 5 predictions for {image_path}:")
    for label, prob in top5_labels:
        print(f"{label}: {prob:.4f}")
    
    predicted_label = labels[top5_catid[0].item()]
    return LABELS_MAP.get(predicted_label, "ostalo")

# Glavna funkcija za klasifikaciju slika u folderima
def classify_images_in_folders(base_dir, labels):
    model = load_model()
    results = {}
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    label = classify_image(model, img_path, labels)
                    # Kreiraj direktorijum ako ne postoji
                    target_folder = os.path.join(base_dir, label)
                    os.makedirs(target_folder, exist_ok=True)
                    # Premesti sliku u odgovarajući folder
                    target_path = os.path.join(target_folder, img_name)
                    os.rename(img_path, target_path)
                    results[target_path] = label
                    print(f"Image {img_path} classified as {label}")
    return results

# Pozovi glavnu funkciju sa direktorijumom koji sadrži slike
base_dir = 'property_images'
labels = load_labels()
results = classify_images_in_folders(base_dir, labels)

# Sačuvaj rezultate u CSV fajl
with open('classification_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['image_path', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for img_path, label in results.items():
        writer.writerow({'image_path': img_path, 'label': label})

print("Classification completed. Results saved to classification_results.csv")
