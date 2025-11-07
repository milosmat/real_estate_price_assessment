import pandas as pd
import os
import glob

# Učitavanje CSV fajla
data = pd.read_csv('property_data.csv')

# Uklanjanje duplikata
data = data.drop_duplicates()

# Uklanjanje stanova bez kvadrature ili cene
data = data.dropna(subset=['kvadratura', 'cena'])

# Pronađite sve ID-jeve koje treba zadržati
valid_ids = set(data['id'])

# Funkcija za brisanje slika na osnovu ID-ja
def delete_images(image_dir, valid_ids):
    for folder in ['kupatilo', 'kuhinja', 'sobe', 'terasa']:
        folder_path = os.path.join(image_dir, folder)
        for img_file in glob.glob(os.path.join(folder_path, '*.jpg')):
            img_id = os.path.basename(img_file).split('_')[0]
            if img_id not in valid_ids:
                os.remove(img_file)
                print(f"Deleted image {img_file}")

# Definisanje direktorijuma sa slikama
image_dir = 'property_images'

# Brisanje slika
delete_images(image_dir, valid_ids)

# Sačuvajte filtrirani CSV fajl
data.to_csv('filtered_property_data.csv', index=False)

print("Filtriranje i brisanje slika završeno.")
