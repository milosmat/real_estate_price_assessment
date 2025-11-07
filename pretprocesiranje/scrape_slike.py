import requests
from bs4 import BeautifulSoup
import csv
import time
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import io

# List of image URLs to block
block_list = [
    "https://resizer2.4zida.rs/unsigned/rs:auto:640:0/plain/https%3A%2F%2Fcdn.4zida.rs%2Fassets%2Fimages%2Fbackgrounds%2Fparkovi.jpg@webp",
    "https://resizer2.4zida.rs/unsigned/rs:auto:640:0/plain/https%3A%2F%2Fcdn.4zida.rs%2Fassets%2Fimages%2Fbackgrounds%2Fkultura.jpg@webp",
    "https://resizer2.4zida.rs/unsigned/rs:auto:640:0/plain/https%3A%2F%2Fcdn.4zida.rs%2Fassets%2Fimages%2Fbackgrounds%2Fobrazovanje.jpg@webp"
]

# Function to get image URLs from a given property URL
def get_image_urls(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        property_id = url.split('/')[-1]  # Extract property ID from URL

        image_urls = []
        img_tags = soup.find_all('img')
        for i, img in enumerate(img_tags):
            img_url = img.get('srcset')
            if img_url:
                img_url = img_url.split(' ')[0]
            else:
                img_url = img.get('src')
            if img_url and not any(blocked in img_url for blocked in block_list):
                img_id = f"{property_id}_{i+1}"
                image_urls.append({
                    "img_id": img_id,
                    "url": img_url
                })

        return property_id, image_urls

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None, []
    except Exception as e:
        print(f"Error processing the URL: {url}, error: {e}")
        return None, []

# Function to get all property links from a range of pages
def get_all_links(start_page, end_page):
    links = []

    for page in range(start_page, end_page + 1):
        try:
            next_page = f'https://www.4zida.rs/prodaja-stanova?strana={page}'
            print(f"Fetching page: {next_page}")
            response = requests.get(next_page)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            property_divs = soup.find_all('div', class_='flex w-2/3 flex-col justify-between py-2')
            for div in property_divs:
                link = div.find('a')['href']
                if 'prodaja-stanova' in link:
                    full_link = 'https://www.4zida.rs' + link
                    if full_link not in links:  # Proveri duplikate pre dodavanja
                        links.append(full_link)

            # Pauza da se izbegne preopterećenje servera
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching links from {next_page}: {e}")
            break

    return links

# Function to check image dimensions
def is_large_image(img_data):
    image = Image.open(io.BytesIO(img_data))
    width, height = image.size
    return width > 300 and height > 300

# Function to download images
def download_images(property_id, image_data):
    image_folder = os.path.join('property_images', property_id)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    for img in image_data:
        img_url = img["url"]
        img_id = img["img_id"]
        if img_url:  # Only download if the URL is not empty
            try:
                img_data = requests.get(img_url).content
                if is_large_image(img_data):  # Check if the image is large enough
                    image_file = io.BytesIO(img_data)
                    image = Image.open(image_file).convert("RGB")
                    file_path = Path(image_folder) / f"{img_id}.jpg"
                    image.save(file_path, "JPEG", quality=80)
                    print(f"Downloaded image {img_id} from {img_url}")
                else:
                    print(f"Skipped small image {img_id} from {img_url}")
            except Exception as e:
                print(f"Error downloading image {img_url}: {e}")

def get_property_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        property_id = url.split('/')[-1]  # Extract property ID from URL

        data = {
            "id": property_id,
            "grad": None,
            "opstina": None,
            "kvart": None,
            "kvadratura": None,
            "broj_soba": None,
            "spratnost": None,
            "stanje": None,
            "grejanje": None,
            "cena": None,
            "lift": None,
            "podrum": None,
            "terasa": None,
            "slike": []
        }

        # Grad, opština i kvart
        location_tag = soup.find('div', class_='flex flex-col items-start justify-center desk:flex-row desk:justify-start desk:gap-2')
        if location_tag:
            spans = location_tag.find_all('span')
            if len(spans) >= 1:
                location_info = spans[0].text.strip().split(', ')
                if len(location_info) >= 3:
                    data["grad"] = location_info[0]
                    data["opstina"] = location_info[1]
                    data["kvart"] = location_info[2]
                elif len(location_info) == 2:
                    data["grad"] = location_info[0]
                    data["opstina"] = location_info[1]

        # Cena
        price_tag = soup.find('div', class_='w-3/8 flex-shrink-0 overflow-clip text-right')
        if price_tag:
            price_p = price_tag.find('p')
            if price_p:
                data["cena"] = price_p.text.strip().replace('€', '').replace('.', '').strip()

        # Kvadratura, broj soba, spratnost
        details_tags = soup.find_all('div', class_='flex flex-1 items-center justify-center bg-white px-2 py-4')
        if details_tags and len(details_tags) >= 3:
            data["kvadratura"] = details_tags[0].find('strong').text.strip().replace('m²', '').strip()
            data["broj_soba"] = details_tags[1].find('strong').text.strip()
            data["spratnost"] = details_tags[2].find('strong').text.strip()

        # O stanu
        stan_tags = soup.select('section.flex.flex-col.gap-1')
        if len(stan_tags) > 0:
            span_tags = stan_tags[0].find_all('span')
            if span_tags:
                for span in span_tags:
                    if 'grejanje' in span.text.lower():
                        data["grejanje"] = span.text.strip()
                    if 'renovirano' in span.text.lower():
                        data["stanje"] = span.text.strip()
                    if 'terasa' in span.text.lower():
                        data["terasa"] = 'Da'
                    if 'podrum' in span.text.lower():
                        data["podrum"] = 'Da'

        # O zgradi
        if len(stan_tags) > 1:
            zgrada_tags = stan_tags[1].find_all('span')
            if zgrada_tags:
                for span in zgrada_tags:
                    if 'lift' in span.text.lower():
                        data["lift"] = 'Da' if '1' in span.text else 'Ne'

        # Preuzimanje slika
        property_id, image_urls = get_image_urls(url)
        data["slike"] = image_urls

        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None
    except Exception as e:
        print(f"Error processing the URL: {url}, error: {e}")
        return None

# Učitavanje već skinutih
# Function to fetch existing IDs from CSV file
def load_existing_ids(csv_file):
    if not os.path.exists(csv_file):
        return set()
    existing_data = pd.read_csv(csv_file)
    return set(existing_data['id'].astype(str).tolist())

# Initialize CSV file with headers if it doesn't exist
csv_file = 'property_data.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "id", "grad", "opstina", "kvart", "kvadratura", "broj_soba", "spratnost", "stanje", 
            "grejanje", "cena", "lift", "podrum", "terasa"
        ])
        writer.writeheader()

# Load existing IDs
existing_ids = load_existing_ids(csv_file)
print(f"Loaded {len(existing_ids)} existing property IDs.")

# Function to check if property ID already exists
def property_exists(property_id, existing_ids):
    return property_id in existing_ids

# Fetch property links from page 1 to 100
property_links = get_all_links(1, 100)

# Scrape data for each property and write to CSV
scraped_properties = set()
for i, link in enumerate(property_links):
    try:
        property_id = link.split('/')[-1]
        if property_exists(property_id, existing_ids):
            print(f"Property ID {property_id} already exists. Skipping.")
            continue
        
        print(f"Fetching property data from: {link}")
        property_data = get_property_data(link)
        if property_data:
            data_tuple = tuple((k, v) for k, v in property_data.items() if k != "slike")  # Exclude images from duplicate check
            if data_tuple not in scraped_properties:  # Proveri duplikate pre dodavanja
                scraped_properties.add(data_tuple)
                print(property_data)  # Print data for verification
                with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=[
                        "id", "grad", "opstina", "kvart", "kvadratura", "broj_soba", "spratnost", "stanje", 
                        "grejanje", "cena", "lift", "podrum", "terasa"
                    ])
                    writer.writerow({k: v for k, v in property_data.items() if k != "slike"})
                download_images(property_data["id"], property_data["slike"])  # Sačuvaj slike

        # Pauza da se izbegne preopterećenje servera
        time.sleep(1)

    except Exception as e:
        print(f"Error fetching property data from {link}: {e}")
        continue

    # Sačuvaj progres svakih 100 zapisa
    if i % 100 == 0:
        print(f"Saved {i} records.")
