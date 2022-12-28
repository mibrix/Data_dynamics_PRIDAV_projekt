import requests
from bs4 import BeautifulSoup
import sqlite3
import os
import csv

path = './faces/'

# Making a GET request
primary = 'https://fmph.uniba.sk'
next = '/pracoviska'

r = requests.get(primary + next)
conn = sqlite3.connect('database.db')
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS persons")
c.execute('''
          CREATE TABLE persons
          (id INTEGER PRIMARY KEY AUTOINCREMENT , meno TEXT, katedra TEXT , oddelenie TEXT, funkcia TEXT )
          ''')

# Parsing the HTML
soup = BeautifulSoup(r.content, 'html.parser')
s = soup.find('div', id= 'c34268').findChildren('a')
katedra = None
s = s[:3] + s[4:]
count_f = 0
count_t = 0
file = open('fakulty.csv','w',encoding='utf-8')
writer = csv.writer(file)
for _ in s:

    katedra = _.text
    next = _['href']

    r = requests.get(primary + next)
    page =  BeautifulSoup(r.content, 'html.parser')
    page = page.select('.bw-borderless tbody tr')
    oddelenie = None
    funkcia = None
    meno = None
    for tag in page:
        if katedra == 'Dekanát':
            try:
                tag['valign']
                if tag.select('br'):
                    funkcia = None

                m = tag.select('a')[0]
                meno = m.text
                link = m['href']

                if os.path.exists(path + link.split('/')[-1].strip() + '.jpg'):
                    c.execute('''
                    INSERT INTO persons (meno,katedra,oddelenie,funkcia)
                    VALUES(?,?,?,?); 
                    ''', (meno, katedra, oddelenie, funkcia))
                    writer.writerow((meno, katedra, oddelenie, funkcia))
                    count_t += 1
                else:
                    count_f += 1
            except KeyError:
                if tag.select('h3'):
                    oddelenie = tag.select('h3')[0].text
                    funkcia = None
                else:
                    if not tag.select('a'):
                        funkcia = tag.text.strip()[:-1]


        else:

            if tag.select('br'):
                oddelenie = None
                funkcia = None

            if len(tag.select('td')) == 1:
                if tag.select('h3'):

                    oddelenie = tag.select('h3')[0].text
                    funkcia = None
                else:
                    if len(tag.select('a')) == 0 :
                        funkcia = tag.text.strip()[:-1]
            if tag.select('a'):
                if tag.select('a')[0]['href'][:10] != "javascript":
                    meno = tag.select('a')[0].text
                    link = tag.select('a')[0]['href']
                    if os.path.exists(path + link.split('/')[-1].strip() + '.jpg'):
                        c.execute('''
                        INSERT INTO persons (meno,katedra,oddelenie,funkcia)
                        VALUES(?,?,?,?); 
                        ''',(meno,katedra,oddelenie,funkcia))
                        writer.writerow((meno, katedra, oddelenie, funkcia))
                        count_t += 1
                    else:
                        count_f += 1




print(count_t,count_f)
conn.commit()