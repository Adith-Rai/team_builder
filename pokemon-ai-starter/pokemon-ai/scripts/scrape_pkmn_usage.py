import os

import requests
from bs4 import BeautifulSoup


def scrape_pkmn_usage_data(base_url="https://www.smogon.com/stats/", save_path="./raw_data/pokemon_usage/"):
    
    base_url = base_url.strip()
    save_path = save_path.strip()
    if base_url[-1] != '/':
        base_url = base_url + '/'
    if save_path[-1] != '/':
        save_path = save_path + '/'
        
        
    url_txt = requests.get(base_url)

    html = BeautifulSoup(url_txt.text, "html.parser")
    html = html.find('pre')

    flag = 0
    for a in html.find_all('a', href=True):
        
        #skip first href
        if flag == 0:
            flag = 1
            continue
        
        print('Season URL:', base_url + a['href'] + '/moveset' )
        season = requests.get(base_url + a['href'] + '/moveset')
        
        formats = BeautifulSoup(season.text, "html.parser")
        
        flag_2 = 0
        for txt in formats.find_all('a', href=True):
            
            #skip first href
            if flag_2 == 0:
                flag_2 = 1
                continue
            
            print('Format URL:', base_url + a['href'] + '/moveset/' + txt['href'])
            usage = requests.get(base_url + a['href'] + '/moveset/' + txt['href'])
            
            if not os.path.exists(save_path + a['href']):
                os.makedirs(save_path + a['href'])
            
            with open(save_path + a['href'] + '/' + txt['href'], 'w+') as fh:
                fh.write(usage.text)


def main():
    
    base_url = "https://www.smogon.com/stats/"

    save_path = "./raw_data/pokemon_usage/"
    
    scrape_pkmn_usage_data(base_url, save_path)
    
    return 'done'
    
    
if __name__ == "__main__":
    
    done = main()
    print(done)