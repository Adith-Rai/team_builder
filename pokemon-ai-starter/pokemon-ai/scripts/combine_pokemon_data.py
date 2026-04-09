import pandas as pd



#Create string dictionary with pokemon data
def parse_text_pkmn_pbs_data(pkmn_pbs_path = r"./raw_data/pokemon/PokemonEssentialsV21.1/PokemonEssentialsV21.1/PBS/pokemon.txt", atts = ['TITLE','NAME', 'TYPES', 'BASESTATS', 'GENDERRATIO', 'GROWTHRATE', 'BASEEXP', 'EVS', 'CATCHRATE', 'HAPPINESS', 'ABILITIES', 'HIDDENABILITIES', 'MOVES', 'TUTORMOVES', 'EGGMOVES', 'EGGGROUPS', 'HATCHSTEPS', 'HEIGHT', 'WEIGHT', 'COLOR', 'SHAPE', 'HABITAT', 'CATEGORY', 'POKEDEX', 'GENERATION', 'EVOLUTIONS', 'WILDITEMUNCOMMON', 'OFFSPRING', 'WILDITEMCOMMON', 'WILDITEMRARE', 'FLAGS', 'FORMNAME', 'INCENSE']):

    text = open(pkmn_pbs_path, "r")
    pkmn_list = text.read().split('#-------------------------------')[1:]

    pokemon_data = []
    for pkmn in pkmn_list:
        
        pokemon_text = list(filter(bool, [line.strip() for line in pkmn.splitlines()]))
        
        if pokemon_text[0].startswith('[') and pokemon_text[0].endswith(']'):
                pokemon_text[0] = "TITLE = " + pokemon_text[0][1:-1].split(',')[0]
        
        pokemon = dict.fromkeys(atts)
        for pkmn in pokemon_text:
            
            line = [p.strip().upper() for p in pkmn.split('=')]
            
            if line[0] not in atts:
                atts = atts + [line[0]]
                print('added new attribute:', line[0])               
            pokemon[line[0]] = line[1]
            
        pokemon_data.append(pokemon)
     
    df = pd.DataFrame(pokemon_data)
    
    return pokemon_data, df


#Combine pokemon forms with original
def combine_parsed_forms_pbs(pkmn_txt_data, forms_txt_data):


    count = 0
    for form in forms_txt_data:
        for pkmn in pkmn_txt_data:
            
            if form['TITLE'].strip() == pkmn['TITLE'].strip():
                
                count += 1
                for key, value in form.items():
                    if key in pkmn.keys() and (value is None or value.strip() == ''):
                        form[key] = pkmn[key]
            
    print(count, len(forms_txt_data))
    
    return pkmn_txt_data + forms_txt_data, pd.DataFrame(pkmn_txt_data + forms_txt_data)                 
    
    
#Combine and Save all raw text data
def combine_parsed_pkmn_pbs_txt(pkmn_pbs_paths = [r"./raw_data/pokemon/PokemonEssentialsV21.1/PokemonEssentialsV21.1/PBS/pokemon.txt", r"./raw_data/pokemon/ScarletViolet_PBS/pokemon.txt"], forms_pbs_paths = [r"./raw_data/pokemon/PokemonEssentialsV21.1/PokemonEssentialsV21.1/PBS/pokemon_forms.txt", r"./raw_data/pokemon/ScarletViolet_PBS/pokemon_forms.txt"], save_path = r"./raw_data/pokemon/intermidiate_pkmn_data/", atts = ['TITLE', 'NAME', 'TYPES', 'BASESTATS', 'GENDERRATIO', 'GROWTHRATE', 'BASEEXP', 'EVS', 'CATCHRATE', 'HAPPINESS', 'ABILITIES', 'HIDDENABILITIES', 'MOVES', 'TUTORMOVES', 'EGGMOVES', 'EGGGROUPS', 'HATCHSTEPS', 'HEIGHT', 'WEIGHT', 'COLOR', 'SHAPE', 'HABITAT', 'CATEGORY', 'POKEDEX', 'GENERATION', 'EVOLUTIONS', 'WILDITEMUNCOMMON', 'OFFSPRING', 'WILDITEMCOMMON', 'WILDITEMRARE', 'FLAGS', 'FORMNAME', 'INCENSE']):
    
    pkmn_data = []
    pkmn_df = pd.DataFrame()
    
    for i in range(len(pkmn_pbs_paths)):
        data, df = parse_text_pkmn_pbs_data(pkmn_pbs_paths[i], atts)
        pkmn_data = pkmn_data + data
        
        if len(forms_pbs_paths) > i:
            forms_data, _ = parse_text_pkmn_pbs_data(forms_pbs_paths[i], atts)
            pkmn_data, df = combine_parsed_forms_pbs(pkmn_data, forms_data)
        
        pkmn_df = pd.concat([pkmn_df, df], ignore_index=True)
        
    pkmn_df.to_csv(save_path + 'raw_pkmn_pbs_data.csv')
    
    return pkmn_data, pkmn_df
        

#Convert text data to usable data structures
def pre_process_useful_pbs_data(pokemon_data):

    pre_processed_data = []
    for pkmn in pokemon_data:

        data = {}
        
        #name
        data['TITLE'] = pkmn['TITLE']
        
        #name
        data['NAME'] = pkmn['NAME']
        
        #form name
        #name
        data['FORMNAME'] = pkmn['FORMNAME']
        
        #types
        types = [p.strip().upper() for p in pkmn['TYPES'].split(',')]
        data['TYPES'] = types
        
        #evolutions
        if pkmn['EVOLUTIONS'] is not None:
            split = pkmn['EVOLUTIONS'].split(',')
            #next_evolutions = [(split[i], split[i+1], split[i+2]) for i in range(0,len(split),3)]
            next_evolutions = [p.strip().upper() for p in split[::3]]
            data['EVOLUTIONS'] = next_evolutions
        else:
            data['EVOLUTIONS'] = []
        
        #stats
        stats = [int(p.strip()) for p in pkmn['BASESTATS'].split(',')]
        stats = stats + [stats[3]]
        del stats[3]
        data['BASESTATS'] = stats
        
        """
        #EVs
        stats_key = {'HP':0, 'ATTACK':1, 'DEFENSE':2, 'SPECIAL_ATTACK':3, 'SPECIAL_DEFENSE':4, 'SPEED':5}
        EVs_list = [p.strip().upper() for p in pkmn['EVS'].split(',')]
        EVs = [0]*6
        for i in range(0,len(EVs_list),2):
            EVs[EVs_list[i]] = int(EVs_list[i+1])
        data[EVS] = EVs
        """
        
        #abilities
        abilities = []
        if pkmn['ABILITIES'] is not None:
            abilities = [p.strip().upper() for p in pkmn['ABILITIES'].split(',')]
        if pkmn['HIDDENABILITIES'] is not None:
            abilities = abilities + [p.strip().upper() for p in pkmn['HIDDENABILITIES'].split(',')]
        data['ABILITIES'] = abilities
        
        #moves 
        """
        split = pkmn['MOVES'].split(',')
        learn_moves = [(int(split[i]), split[i+1]) for i in range(0, len(split),2)]
        """
        moves = [x.strip().upper() for x in pkmn['MOVES'].split(',') if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
        tutor_moves = []
        if pkmn['TUTORMOVES'] is not None:
            tutor_moves = [p.strip().upper() for p in pkmn['TUTORMOVES'].split(',')]
        egg_moves = []
        if pkmn['EGGMOVES'] is not None:
            egg_moves = [p.strip().upper() for p in pkmn['EGGMOVES'].split(',')]
        data['MOVES'] = list(dict.fromkeys(egg_moves + moves + tutor_moves))
        
        #weight
        data['WEIGHT'] = float(pkmn['WEIGHT'].strip())
        
        #generation
        data['GENERATION'] = int(pkmn['GENERATION'].strip())
        
        pre_processed_data.append(data)
    
    #Map moves to pokemon evolution lines
    for pkmn in pre_processed_data:
    
        evo = pkmn
      
        for i in range(20):
        
            if i>3:
                print("Exceeded more than 3 evolutions for:", evo['TITLE'])
        
            if evo['EVOLUTIONS'] is None or evo['EVOLUTIONS'] == []:
                break
            
            tmp = pkmn
            for data in pre_processed_data:
                if data['NAME'] in evo['EVOLUTIONS']:
                    comb = evo['MOVES'] + data['MOVES']
                    data['MOVES'] = list(dict.fromkeys(comb))
                    tmp = data
            evo = tmp
            
    df = pd.DataFrame(pre_processed_data)
            
    return pre_processed_data, df
                    
                    
                
            
            
            
            
            
            
            
        
        
    
def main():
    
    pkmn_gen1to8_pbs_path = r"./raw_data/pokemon/PokemonEssentialsV21.1/PokemonEssentialsV21.1/PBS/pokemon.txt"
    pkmn_gen9_pbs_path = r"./raw_data/pokemon/ScarletViolet_PBS/pokemon.txt"
    
    forms_gen1to8_pbs_path = r"./raw_data/pokemon/PokemonEssentialsV21.1/PokemonEssentialsV21.1/PBS/pokemon_forms.txt" 
    forms_gen9_pbs_path = r"./raw_data/pokemon/ScarletViolet_PBS/pokemon_forms.txt"

    save_path = r"./raw_data/pokemon/intermidiate_pkmn_data/"
    
    atts = ['TITLE', 'NAME', 'TYPES', 'BASESTATS', 'GENDERRATIO', 'GROWTHRATE', 'BASEEXP', 'EVS', 'CATCHRATE', 'HAPPINESS', 'ABILITIES', 'HIDDENABILITIES', 'MOVES', 'TUTORMOVES', 'EGGMOVES', 'EGGGROUPS', 'HATCHSTEPS', 'HEIGHT', 'WEIGHT', 'COLOR', 'SHAPE', 'HABITAT', 'CATEGORY', 'POKEDEX', 'GENERATION', 'EVOLUTIONS', 'WILDITEMUNCOMMON', 'OFFSPRING', 'WILDITEMCOMMON', 'WILDITEMRARE', 'FLAGS', 'FORMNAME', 'INCENSE']

    text_data, df = combine_parsed_pkmn_pbs_txt([pkmn_gen1to8_pbs_path, pkmn_gen9_pbs_path], [forms_gen1to8_pbs_path, forms_gen9_pbs_path], save_path, atts)
    
    pre_processed_data, df = pre_process_useful_pbs_data(text_data)
    df.to_csv(save_path + 'pkmn_pbs_data.csv')
    
    """
    for i in pre_processed_data:
        if i['TITLE'].startswith('PORYGON'):
            print(i['NAME'] + ':',len(i['MOVES']))
    """
    

    
main()
            

    
    
    

