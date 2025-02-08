import numpy as np
import colorama as cr


def generate_map(width,mine_freq):
    map = np.zeros((width,width))
    for yiter in range(0,width):
        for xiter in range(0,width):
            if np.random.rand() < mine_freq:
                map[yiter,xiter] = 9
                if yiter != 0:
                    if map[yiter-1,xiter] != 9:
                        map[yiter-1,xiter] = map[yiter-1,xiter] + 1
                    if xiter != 0:
                        if map[yiter-1,xiter-1] != 9:
                            map[yiter-1,xiter-1] = map[yiter-1,xiter-1] + 1
                    if xiter != width-1:
                        if map[yiter-1,xiter+1] != 9:
                            map[yiter-1,xiter+1] = map[yiter-1,xiter+1] + 1
                if xiter != 0:
                    if map[yiter,xiter-1] != 9:
                        map[yiter,xiter-1] = map[yiter,xiter-1] + 1
                if yiter != width-1:
                    if map[yiter+1,xiter] != 9:
                        map[yiter+1,xiter] = map[yiter+1,xiter] + 1
                    if xiter != 0:
                        if map[yiter+1,xiter-1] != 9:
                            map[yiter+1,xiter-1] = map[yiter+1,xiter-1] + 1
                    if xiter != width-1:
                        if map[yiter+1,xiter+1] != 9:
                            map[yiter+1,xiter+1] = map[yiter+1,xiter+1] + 1
                if xiter != width-1:
                    if map[yiter,xiter+1] != 9:
                        map[yiter,xiter+1] = map[yiter,xiter+1] + 1
    return map

def print_map(map, opened:list):
    for y in range(0,len(map[0])):
        for x in range(0,len(map)):
            print((cr.Fore.BLACK if (y,x) not in opened and (y,x) not in flags
                   else (cr.Fore.MAGENTA if (y,x) in flags
                         else (cr.Fore.RESET if map[y,x] in [0,1,2,3,4,5,6,7,8]
                               else cr.Fore.RED))) + str(int(map[y,x])), end=" ")
        print(cr.Fore.RESET + "\n",end="")

#fixed-to-the-main-cycle functions
excluded = []
def recursive_open(tile_coords):
    if map[tile_coords] == 0 and not tile_coords in excluded:
        excluded.append(tile_coords)
        for i in get_adjacent(tile_coords):
            if i != "OoB":
                recursive_open(i)
    if tile_coords not in opened:
        opened.append(tile_coords)
def empty_recursive_cache():
    global excluded
    excluded = []

def get_adjacent(tile_coords):
    result = []
    result.append("OoB" if tile_coords[0] == 0 or tile_coords[1] == 0
                  else (tile_coords[0]-1, tile_coords[1]-1))
    result.append("OoB" if tile_coords[0] == 0
                  else (tile_coords[0]-1, tile_coords[1]))
    result.append("OoB" if tile_coords[0] == 0 or tile_coords[1] == map_width-1
                  else (tile_coords[0]-1, tile_coords[1]+1))
    result.append("OoB" if tile_coords[1] == 0
                  else (tile_coords[0], tile_coords[1]-1))
    result.append("OoB" if tile_coords[1] == map_width-1
                  else (tile_coords[0], tile_coords[1]+1))
    result.append("OoB" if tile_coords[0] == map_width-1 or tile_coords[1] == 0
                  else (tile_coords[0]+1, tile_coords[1]-1))
    result.append("OoB" if tile_coords[0] == map_width-1
                  else (tile_coords[0]+1, tile_coords[1]))
    result.append("OoB" if tile_coords[0] == map_width-1 or tile_coords[1] == map_width-1
                  else (tile_coords[0]+1, tile_coords[1]+1))
    return result

def all_available():
    available = set()
    for coords in opened:
        for i in get_adjacent(coords):
            if i not in opened and i != "OoB":
                available.add(i)
    return available

def get_most_known():
    knowledge_list = np.array([["0,0","-1"]])
    for coords in all_available():
        knowledge = 0
        for adj in get_adjacent(coords):
            if adj == "OoB":
                knowledge += 1
            if adj in opened:
                if adj not in flags:
                    knowledge += 3
                    if map[adj] in [1,2]:
                        knowledge += 4-map[adj]
                else:
                    knowledge += 2
        knowledge += 4 * count_open_corner(coords)
        for tile in get_adjacent(coords):
            if tile in opened and tile not in flags:
                local_km = 0
                for i in get_adjacent(tile):
                    if i in flags:
                        local_km += 1
                if local_km == map[tile]:
                    knowledge += 6
        knowledge_list = np.concatenate((knowledge_list, np.array([[f"{coords[0]},{coords[1]}", str(knowledge)]])), axis=0)
    # return tuple([int(j) for j in knowledge_list[knowledge_list[:,1].tolist().index(str(max([int(i) for i in knowledge_list[:,1]]))), 0].split(",")])
    return knowledge_list

def get_slice(coords, hide_flags = False):
    nums = []
    kms = []
    unkws = []
    for adj_crds in get_adjacent(coords):
        local_km = 0
        local_unkw = 0
        if adj_crds == "OoB":
            nums.append(11)
            kms.append(11)
            unkws.append(11)
        else:
            if adj_crds in opened:
                for i in get_adjacent(adj_crds):
                    if i in flags:
                        local_km += 1
                    if i not in opened and i != "OoB":
                        local_unkw += 1
                if adj_crds not in flags:
                    nums.append(int(map[adj_crds]))
                    kms.append(local_km)
                    unkws.append(local_unkw)
                else:
                    nums.append(9 if not hide_flags else 10)
                    kms.append(9 if not hide_flags else 10)
                    unkws.append(9 if not hide_flags else 10)
            else:
                nums.append(10)
                kms.append(10)
                unkws.append(10)
    return nums + kms + unkws

def count_open_corner(coords):
    adj_list = get_adjacent(coords)
    corner_indices = [[0,1,3], [1,2,4], [3,5,6], [4,6,7]]
    corner_count = 0
    for i in corner_indices:
        corner_similarity = 0
        for j in i:
            if adj_list[j] in opened:
                corner_similarity += 1
        if corner_similarity == 3:
            corner_count += 1
    return corner_count

opened = []
knowledge = []
flags = []
map_width = 9
map = np.zeros((map_width,map_width))



def simulate_data(data_amount, pass_freq):
    print("Generating data...")
    global map
    global opened
    global knowledge
    global flags
    amount = 0
    first_count = 0
    second_count = 0
    _already_printed = False
    mine_freq = 0.12
    while amount < data_amount:
        opened = []
        knowledge = []
        flags = []
        iteration = 0
        map = generate_map(map_width,mine_freq)

        pick = (np.random.randint(0,map_width-1),np.random.randint(0,map_width-1))
        while map[pick] == 9:
            pick = (np.random.randint(0,map_width-1),np.random.randint(0,map_width-1))
        recursive_open(pick)
        empty_recursive_cache()


        while len(opened) < map_width**2:
            empty_recursive_cache()
            if amount == int(data_amount*0.9):
                if not _already_printed:
                    print(cr.Fore.GREEN + "<||||90|||.>" + cr.Fore.RESET)
                    _already_printed = True
                mine_freq = 0.05
            elif amount == int(data_amount*0.8):
                if not _already_printed:
                    print(cr.Fore.GREEN + "<||||80||..>" + cr.Fore.RESET)
                    _already_printed = True
                mine_freq = 0.3
            elif amount == int(data_amount*0.7):
                if not _already_printed:
                    print(cr.Fore.GREEN + "<||||70|...>" + cr.Fore.RESET)
                    _already_printed = True
                mine_freq = 0.25
            elif amount == int(data_amount*0.6):
                if not _already_printed:
                    print(cr.Fore.GREEN + "<||||60....>" + cr.Fore.RESET)
                    _already_printed = True
                mine_freq = 0.2
            elif amount == int(data_amount*0.5):
                if not _already_printed:
                    print(cr.Fore.GREEN + "<||||50....>" + cr.Fore.RESET)
                    _already_printed = True
                mine_freq = 0.15
            elif amount == int(data_amount*0.4):
                if not _already_printed:
                    print(cr.Fore.GREEN + "<||||40....>" + cr.Fore.RESET)
                    _already_printed = True
                mine_freq = 0.1
            elif amount == int(data_amount*0.3):
                if not _already_printed:
                    print(cr.Fore.GREEN + "<|||.30....>" + cr.Fore.RESET)
                    _already_printed = True
            elif amount == int(data_amount*0.2):
                if not _already_printed:
                    print(cr.Fore.GREEN + "<||..20....>" + cr.Fore.RESET)
                    _already_printed = True
            elif amount == int(data_amount*0.1):
                if not _already_printed:
                    print(cr.Fore.GREEN + "<|...10....>" + cr.Fore.RESET)
                    _already_printed = True
            iteration += 1
            if len(opened) > 3:
                knowledges = get_most_known()
                pick = tuple([int(i) for i in knowledges[np.array([float(i) for i in knowledges[:, 1]]).argsort()[-1], 0].split(",")])
            else:
                pick = list(all_available())[np.random.randint(0,len(all_available()))]
            slice = get_slice(pick)
            decision = np.random.choice([1,2,2])

            best_option = decision
            if decision == 1:
                if map[pick] == 9:
                    best_option = 2
                    decision = 2
                else:
                    best_option = 1
                    recursive_open(pick)
                    empty_recursive_cache()
            if decision == 2:
                if map[pick] != 9:
                    best_option = 1
                    recursive_open(pick)
                    empty_recursive_cache()
                else:
                    flags.append(pick)
                    opened.append(pick)

            if (np.random.random() < pass_freq or (map[pick] == 9 and np.random.random() < pass_freq*1.3)
                    or (first_count >= 0.5 * data_amount and map[pick] == 9 and np.random.random() < pass_freq*2)):
                if amount < data_amount:
                    if best_option == 1:
                        if first_count <= 0.5 * data_amount:
                            yield (slice, best_option)
                            first_count += 1
                            amount += 1
                            _already_printed = False
                    else:
                        if second_count <= 0.5 * data_amount:
                            yield (slice, best_option)
                            second_count += 1
                            amount += 1
                            _already_printed = False
    print(cr.Fore.GREEN + "<||||100|||>" + cr.Fore.RESET)