import pandas as pd
import requests


class Pokemon:
    def __init__(self, name, abilities, items, spreads, moves, teammates, counters):
        self.name = name
        self.abilities = self.resolve_abilities(abilities)
        self.items = self.resolve_items(items)
        self.spreads = self.resolve_spreads(spreads)
        self.moves = self.resolve_moves(moves)
        self.teammates = self.resolve_teammates(teammates)
        self.counters = self.resolve_counters(counters)

    def __repr__(self):
        return str(self.as_frame())

    def as_frame(self):
        values = [
            self.name, len(self.abilities), len(self.items),
            len(self.spreads), len(self.moves),
            len(self.teammates), len(self.counters)
        ]
        return pd.DataFrame([values], columns=["NAME", "ABILITIES", "ITEMS", "BUILDS", "MOVES", "TEAMMATES", "COUNTERS"])

    def get_frames(self):
        frames = {
            "abilities": pd.DataFrame([i.as_frame().values[0] for i in self.abilities], columns=["NAME", "RATE"]),
            "moves": pd.DataFrame([i.as_frame().values[0] for i in self.moves], columns=["NAME", "RATE"]),
            "items": pd.DataFrame([i.as_frame().values[0] for i in self.items], columns=["NAME", "RATE"]),
            "spreads": pd.DataFrame([i.as_frame().values[0] for i in self.spreads], columns=["NATURE", "RATE", "HP", "ATK", "DEF", "SPA", "SPD", "SPE"]),
            "teammates": pd.DataFrame([i.as_frame().values[0] for i in self.teammates], columns=["NAME", "RATE"]),
            "counters": pd.DataFrame([i.as_frame().values[0] for i in self.counters], columns=["NAME", "MEAN", "KO%", "SWITCH%"]),
            
        }
        return frames
    
    def resolve_abilities(self, abilities):
        return [Ability(*ability) for ability in abilities]

    def resolve_items(self, items):
        return [Item(*item) for item in items]

    def resolve_spreads(self, spreads):
        return [NatureEffort(*evs) for evs in spreads]

    def resolve_moves(self, moves):
        return [Move(*move) for move in moves]

    def resolve_teammates(self, teammates):
        return [Teammate(*teammate) for teammate in teammates]

    def resolve_counters(self, counters):
        return [CounterPoke(*counter) for counter in counters]


class AbstractAttributeProba:
    def __init__(self, name, rate):
        self.name = name
        self.rate = rate

    def __repr__(self):
        return f"{self.name}: {self.rate}"

    def as_frame(self):
        return pd.DataFrame([[self.name, self.rate]], columns=["NAME", "RATE"])

class Ability(AbstractAttributeProba):
    def __init__(self, name, rate):
        super().__init__(name, rate)


class Item(AbstractAttributeProba):
    def __init__(self, name, rate):
        super().__init__(name, rate)


class NatureEffort(AbstractAttributeProba):
    def __init__(self, name, rate, hp, atk, dfs, spa, spd, spe):
        super().__init__(name, rate)
        self.hp = hp
        self.atk = atk
        self.dfs = dfs
        self.spa = spa
        self.spd = spd
        self.spe = spe

    def __repr__(self):
        return f"| {repr(super()) } -> | HP: {self.hp} | ATK: {self.atk} | DEF: {self.dfs} | SPA: {self.spa} | SPD: {self.spd} | SPE: {self.spe} |"

    def as_frame(self):
        return pd.DataFrame([[
            self.name, self.rate,
            self.hp, self.atk,
            self.dfs, self.spa,
            self.spd, self.spe
        ]],
            columns=["NATURE", "RATE", "HP", "ATK", "DEF", "SPA", "SPD", "SPE"]
        )

    
class Move(AbstractAttributeProba):
    def __init__(self, name, rate):
        super().__init__(name, rate)


class Teammate(AbstractAttributeProba):
    def __init__(self, name, rate):
        super().__init__(name, rate)


class CounterPoke:
    def __init__(self, name, mean_proba, ko_proba, switch_proba):
        self.name = name
        self.mean_proba = mean_proba
        self.ko_proba = ko_proba
        self.switch_proba = switch_proba

    def __repr__(self):
        return f"| {self.name}: Ocurrences: {self.mean_proba} | P(K.O): {self.ko_proba} | P(Switch): {self.switch_proba} |"

    def as_frame(self):
        return pd.DataFrame(
            [[self.name, self.mean_proba, self.ko_proba, self.switch_proba]],
            columns=["NAME", "MEAN", "KO%", "SWITCH%"]
        )


class PokeStats:
    def __init__(self, data, source=None):
        self.poke_tables = {}
        self.pre_process(data)
        self.source = source if source is not None else "Not specified"

    def __len__(self):
        return len(self.poke_tables)

    def __getitem__(self, key):
        return self.resolve_pokemon(key)

    def __repr__(self):
        return f"| Moveset collection | Total objects: {len(self)} | Data source: {self.source} |"
    
    def resolve_pokemon(self, name):
        metadata = self.poke_tables.get(name.title())
        if metadata is None:
            return

        return Pokemon(**metadata)

    def resolve_tables(self):
        return [self.resolve_pokemon(name) for name in self.poke_tables]

    def as_frame(self):
        return pd.DataFrame([i.as_frame().values[0] for i in self.resolve_tables() if i], columns=["NAME", "ABILITIES", "ITEMS", "BUILDS", "MOVES", "TEAMMATES", "COUNTERS"])
    
    def pre_process(self, data):
        # The raw data is a large text with many pokemon tables
        # start spliting each pokemon from the raw content
        # each index is a dirty text table containing the data of a unique pokemon
        raw_mons = data.strip().split("+----------------------------------------+ \n +----------------------------------------+ \n")
        
        # for each pokemon, slice the stacked cells that contains the information
        
        for dirty_table in raw_mons:
            cells = dirty_table.split("\n +----------------------------------------+")

            # each cell is like a transposed table
            name = self.get_pokemon_name(cells[0])
            abilities = self.get_rating_attribute(cells[2])
            items = self.get_rating_attribute(cells[3])

            # preprocess the data first with the get rating so theres no need to duplicate code
            spreads = self.get_spreads(self.get_rating_attribute(cells[4]))

            moves = self.get_rating_attribute(cells[5])
            mates = self.get_rating_attribute(cells[6])
            counters = self.get_counters(cells[7])

            self.poke_tables[name] = {
                "name": name,
                "abilities": abilities,
                "items": items,
                "spreads": spreads,
                "moves": moves,
                "teammates": mates,
                "counters": counters
            }

    def get_pokemon_name(self, cell_header):
        slice_point = cell_header.index("|")
        return cell_header[slice_point:].replace("|", "").strip()

    def get_rating_attribute(self, dirty_cell):
        # start at index 1 to avoid the cell name
        raw_list = dirty_cell.split("\n")[1:]
        
        clean_list = []
        for row in raw_list:
            # the slicing point is two characters before the floating point
            line = row.replace("|", "").strip().replace("%", "")
            
            dot = line.find(".")
            if dot < 0:
                continue

            rate = float(line[dot-3:].strip())
            name = line[:dot-3].strip()
            clean_list.append((name, rate))
            
        return clean_list

    def get_spreads(self, preprocessed_ratings):
        spreads = []
        for preprocessed in preprocessed_ratings:
            dirty, rate = preprocessed
            if dirty == "Other":
                continue
            nature, evs = dirty.split(":")
            efforts = [int(ev) for ev in evs.split("/")]
            spreads.append([nature, rate] + efforts)

        return spreads

    def get_counters(self, dirty_cell):
        raw_list = dirty_cell.strip().split("\n")[1:]
        lines =  ("".join(raw_list)).split("|  |")
        clean_list = []

        for line in lines:
            preproc = line.replace("|", "").replace("%", "").replace(")", "").strip().split("(")
            if len(preproc) < 3:
                continue

            name_mean, _, ko_swt = preproc

            dot = name_mean.find(".")
            mean = float(name_mean[dot-3:].strip())
            name = name_mean[:dot-3].strip()

            ko, swt = ko_swt.split("/")
            ko = float(ko.strip().split(" ")[0])
            swt = float(swt.strip().split(" ")[0])

            clean_list.append((name, mean, ko, swt))
        return clean_list 
		


def fit(model, X, y):
    for comb in X:
        for mon in comb:
            for k, v in y[mon].items():
                if k in model[comb]:
                    model[comb][k] = (model[comb][k] + v) / 2
                else:
                    model[comb][k] = v

    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count

    return model
    


#################################################

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

#################################################

r = requests.get('https://www.smogon.com/stats/2014-11/moveset/ou-1500.txt')
stats = PokeStats(r.text, r.url)
df = stats.as_frame()



"""
cresselia = stats["cresselia"].get_frames()
print(cresselia['spreads'])
print(cresselia['moves'])
"""




from itertools import combinations

# Generate Pokemon combinations
meta = df.NAME.unique()
list_combinations = list()

list_combinations += list(combinations(meta, 1))
list_combinations += list(combinations(meta, 2))






from collections import defaultdict


mates_model = defaultdict(lambda: defaultdict(lambda: 0))

for comb in list_combinations:
    mates_model[comb] = {}

# preproc teammates and counters
mates_map = {}
for mon in df.NAME.values:
    mates_map[mon] = dict(stats[mon].get_frames()['teammates'].values) 
	

mates_model = fit(mates_model, list_combinations, mates_map)


import random


def teambuild(model, team):
    team_finished = False
    team_len = len(team)    
    
    while not team_finished:
      # select a random probability threshold
      r = random.random() + (random.random() * random.random())
      
      accumulator = (random.random() * random.random())
      for mate in model[tuple(team[-2:])].keys():
          accumulator += model[tuple(team[-2:])][mate]
          if accumulator >= r and mate not in team:
              team.append(mate)
              break

      if len(team) == 6:
          team_finished = True
          break
    
      if len(team) > team_len:
          team_len += 1
      else:
          random.shuffle(team)
    
    return team



teams = []
for _ in range(5):
    team = ['Amoonguss']
    gen = teambuild(mates_model, team)
    teams.append(gen)



teamview = pd.DataFrame()
for team in teams:
    rows = []
    team_items = []
    view = pd.DataFrame()
    for mate in team:
        frames = stats[mate].get_frames()
        moves = frames['moves'].NAME.values[:4]
        items = frames['items'][~frames['items'].NAME.isin(team_items)]
        if len(items) < 1:
            item = None
        else:
            item = items.NAME.values[0]

        team_items.append(item)
        row = [mate, item] + list(moves)
        rows.append(row)
    view = pd.DataFrame(rows, columns=['POKE', 'ITEM', 'MOVE1', 'MOVE2', 'MOVE3', 'MOVE4'])
    teamview = pd.concat([teamview, view], ignore_index=True)

print(teamview.head(50))












