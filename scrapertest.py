import requests
import csv
from bs4 import BeautifulSoup

url = "https://www.tasso.net/Tierregister/Suchmeldungen"

response = requests.get(url)

html = BeautifulSoup(response.text, 'html.parser')

t = html.find_all('div', class_="medium-4 columns end small-buffer-half-bottom")
barkers = html.find_all('article', class_="row")


print (barkers)

#print(html)

