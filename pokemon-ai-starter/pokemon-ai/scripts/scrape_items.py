import pandas as pd

import requests
from bs4 import BeautifulSoup
from ftfy import fix_encoding


def clean_text(text):
        no_spec = fix_encoding(text)
        #no_spec = "".join(s for s in no_spec if ord(s)<128)
        split_end = no_spec.split()
        if split_end[-1] != "(R)" and split_end[-1] != "(B)" and split_end[-1] != "(Y)":
            if split_end[-1][0] == '(' and split_end[-1][-1] == ')':
                no_spec = "".join(split_end[:-1])
        return no_spec


#Get all items divided by generation
def get_all_items_by_gen(base_url="https://bulbapedia.bulbagarden.net", page="/wiki/Category:Items_by_generation"):

    print("Scraping Item Generations...")
    #Get links to items per gen
    listOfUrls = requests.get( base_url + page )

    html = BeautifulSoup(listOfUrls.text, "html.parser")
    html = html.find('div', class_='mw-category-group')
    html = html.find('ul')

    #Get all items from each gen
    df = pd.DataFrame({'item_' : [], 'gen_added': []})
    gen = [1,2,3,4,9,5,6,7,8]
    i = 0
    for a in html.find_all('a', href=True):

        items_link = requests.get(base_url + a['href'])
        items = BeautifulSoup(items_link.text, "html.parser")
        items = items.find('div', class_='mw-content-ltr')
        
        for a in items.find_all('a', href=True):
            df = pd.concat([pd.DataFrame([[a.text.strip(), gen[i]]], columns=df.columns), df], ignore_index=True)      
        i += 1
    
    df['item_name'] = df['item_'].apply(clean_text)
    df['item_name'] = df['item_name'].str.lower()
    df['item_name'] = df['item_name'].str.replace(" ", "")
    df = df.drop(columns=['item_'])
    
    print("Scraped Item Generations Sucessfully!")
    return df
    
    
def get_all_item_numbers(base_url="https://bulbapedia.bulbagarden.net", page="/wiki/List_of_items_by_index_number_(Generation_IX)"):
    
    print("Scraping Item numbers...")
    #Item Number URL
    itemNoUrl = requests.get( base_url + page )

    html = BeautifulSoup(itemNoUrl.text, "html.parser")

    df = pd.DataFrame({'item_no' : [], 'hex': [], 'item': [], 'pocket': []})
    
    #Get all item numbers 
    skip = 1
    html = html.find('table', class_='roundy').find('table', class_='roundy sortable')
    for item_row in html.find_all('tr'):
    
        if skip == 1:
            skip = 0
            continue
        
        item_info = item_row.find_all('td')
        
        td_count = len(item_info)
        item_no = 0
        hex_no = '0x0000'
        item = 'None'
        pocket = 'None'
        if td_count >= 1:
            item_no = int(item_info[0].text.strip())
        if td_count >= 2: 
            hex_no = item_info[1].text.strip()
        if td_count >= 4:
            a = item_info[3].find('a')
            if a is not None:
                item = a.text.strip()
            else:
                item = item_info[3].text.strip()
        if td_count >= 6:
            pocket = item_info[5].text.strip()
        
        df = pd.concat([pd.DataFrame([[item_no, hex_no, item, pocket]], columns=df.columns), df], ignore_index=True)
    
    
    df['item_name'] = df['item'].apply(clean_text)
    df['item_name'] = df['item_name'].str.lower()
    df['item_name'] = df['item_name'].str.replace(" ", "")
        
    print("Scraped Item Numbers Sucessfully!")
    return df
        
        
def merge_and_save(item_nos, item_gen, save_path="./raw_data/items"):
    
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    
    if 'item_no' in item_nos.columns and 'gen_added' in item_gen.columns:
        df1 = item_nos
        df2 = item_gen
    elif 'item_no' in item_gen.columns and 'gen_added' in item_nos.columns:
        df1 = item_gen
        df2 = item_nos
    else:
        print("Incorrect scraped data format")
        return pd.DataFrame()
    
    df = df1.merge(df2, on='item_name', how='left')
    df['gen_added'] = df['gen_added'].fillna(value=0)
    df = df.sort_values(by=['item_no', "gen_added"])
    #df = df.drop_duplicates(subset=['item_no', 'item_name'], keep='first')
    
    df.to_csv(save_path + 'items' + '.csv')
    
    print("Merged and saved items data")
    return df


#Scrape all pokemon moves
def get_all_pokemon_moves(base_url="https://bulbapedia.bulbagarden.net", page="/wiki/List_of_moves", save_path="./raw_data/movesets/"):
    
    print("Scraping Pokemon Moves...")
    #Item Number URL
    movesURL = requests.get( base_url + page )

    html = BeautifulSoup(movesURL.text, "html.parser")

    df = pd.DataFrame({'move_no' : [], 'move': [], 'type': [], 'category': [], 'pp': [], 'power': [], 'accuracy': [], 'gen_added': []})
    
    #Get all item numbers 
    skip = 1
    html = html.find('table', class_='sortable roundy').find('table', class_='sortable roundy')
    for move_row in html.find_all('tr'):
    
        if skip == 1:
            skip = 0
            continue
        
        move_info = move_row.find_all('td')
        
        gen_map = {'I':1, 'II':2, 'III':3, 'IV':4, 'V':5, 'VI':6, 'VII':7, 'VIII':8, 'IX':9}
        td_count = len(move_info)
        move_no = 0
        pp = 0
        gen_added = 0
        power = '—'
        accuracy = '—'
        move = 'None'
        type_ = 'None'
        category = 'None'
        
        if td_count >= 1:
            move_no = int(move_info[0].text.strip())
        if td_count >= 2: 
            a = move_info[1].find('a')
            move = a.text.strip()
        if td_count >= 3:
            a = move_info[2].find('a').find('span')
            type_ = a.text.strip()
        if td_count >= 4:
            a = move_info[3].find('a').find('span')
            category = a.text.strip()
        if td_count >= 5:
            pp = int(move_info[4].text.strip())
        if td_count >= 6:
            power = move_info[5].text.strip()
        if td_count >= 7:
            accuracy = move_info[6].text.strip()
        if td_count >= 8:
            a = move_info[7].find('a').find('span')
            if a.text.strip() in gen_map:
                gen_added = gen_map[a.text.strip()]
        
        df = pd.concat([pd.DataFrame([[move_no, move, type_, category, pp, power, accuracy, gen_added]], columns=df.columns), df], ignore_index=True)
    
    
    df['move_name'] = df['move'].apply(clean_text)
    df['move_name'] = df['move_name'].str.lower()
    df['move_name'] = df['move_name'].str.replace(" ", "")
    
    df = df.sort_values(by=['move_no', "gen_added"])
    df.to_csv(save_path + 'moves' + '.csv')
        
    print("Scraped Pokemon Moves Sucessfully!")
    return df  
    

def main():

    url = "https://bulbapedia.bulbagarden.net"  
    
    item_page = "/wiki/Category:Items_by_generation"
    item_no_page = "/wiki/List_of_items_by_index_number_(Generation_IX)"
    items_save_path = "./raw_data/items/"
    
    moves_page="/wiki/List_of_moves"
    moves_save_path = "./raw_data/movesets/"
    
    items_df = get_all_items_by_gen(url, item_page)
    item_nos_df = get_all_item_numbers(url, item_no_page)
    items_df = merge_and_save(item_nos_df, items_df, items_save_path)
    
    moves_df = get_all_pokemon_moves(url, moves_page, moves_save_path)
    
    return items_df, moves_df
    
    
if __name__ == "__main__":

    items_df, moves_df = main()
    #print(df[df['gen_added'] > 0].drop_duplicates(subset=['item_no', 'item_name'], keep='first').shape)
    print(items_df.head(10))
    print(moves_df.head(10))


    



