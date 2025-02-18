import numpy as np
import colorama as cr
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Conv2D, Flatten

from simulator import simulate_data, generate_map

def print_map(map, opened:list):
    for y in range(0,len(map[0])):
        for x in range(0,len(map)):
            print((cr.Fore.BLACK if (y,x) not in opened and (y,x) not in flags
                   else (cr.Fore.MAGENTA if (y,x) in flags
                         else (cr.Fore.RESET if map[y,x] in [0,1,2,3,4,5,6,7,8]
                               else cr.Fore.RED))) + str(int(map[y,x])), end="  ")
        print(cr.Fore.RESET + "\n",end="")

excluded = []
def recursive_open(tile_coords):
    if map[tile_coords] == 0 and not tile_coords in excluded:
        excluded.append(tile_coords)
        for i in get_adjacent(tile_coords):
            if i != "OoB":
                recursive_open(i)
    if tile_coords not in opened:
        opened.append(tile_coords)
    if tile_coords in flags:
        flags.remove(tile_coords)
    if tile_coords in temporal_uncertainty:
        temporal_uncertainty.remove(tile_coords)
    for i in get_adjacent(tile_coords):
        if i in temporal_uncertainty:
            temporal_uncertainty.remove(i)
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
                        knowledge += 3-map[adj]
                else:
                    knowledge += 2
        knowledge += 4 * count_open_corner(coords)
        if count_open_corner(coords) == 0:
            knowledge -= 3
        for tile in get_adjacent(coords):
            if tile in opened and tile not in flags:
                if count_known_mines(tile) == map[tile] or count_unknowns(tile) == map[tile]-count_known_mines(tile):
                    knowledge += 10
        knowledge_list = np.concatenate((knowledge_list, np.array([[f"{coords[0]},{coords[1]}", str(knowledge)]])), axis=0)
    # return tuple([int(j) for j in knowledge_list[knowledge_list[:,1].tolist().index(str(max([int(i) for i in knowledge_list[:,1]]))), 0].split(",")])
    return knowledge_list

def get_slice(coords, hide_flags = False):
    result = np.zeros((7,7))
    for y in range(-3,4):
        for x in range(-3,4):
            ty = coords[0]+y
            tx = coords[1]+x
            tile = (ty,tx)
            if (ty > map_width-1 or ty < 0) or (tx > map_width-1 or tx < 0):
                result[y+3,x+3] = 11
            else:
                if tile in opened:
                    if tile in flags:
                        if hide_flags:
                            result[y+3,x+3] = 10
                        else:
                            result[y+3,x+3] = 9
                    else:
                        result[y+3,x+3] = map[tile]
                else:
                    result[y+3,x+3] = 10
    result[3,3] = 99
    return result
def all_iso_variants(array):
    layout = array[0]
    option = array[1]
    for var in [layout, np.rot90(layout, 1), np.rot90(layout, 2), np.rot90(layout, 3)]:
        for i in [False,True]:
            if i:
                yield (np.fliplr(var),option)
            else:
                yield (var,option)
def not_flags():
    return sum([1 if i not in flags else 0 for i in opened])

def count_open_corner(coords):
    adj_list = get_adjacent(coords)
    corner_indices = [[0,1,3], [1,2,4], [3,5,6], [4,6,7]]
    corner_count = 0
    for i in corner_indices:
        corner_similarity = 0
        for j in i:
            if adj_list[j] in opened and adj_list[j] not in flags:
                corner_similarity += 1
        if corner_similarity == 3:
            corner_count += 1
    return corner_count

def count_known_mines(coords):
    local_mines = 0
    for i in get_adjacent(coords):
        if i in flags:
            local_mines += 1
    return local_mines

def count_unknowns(coords, guessed = False):
    local_unkws = 0
    for i in get_adjacent(coords):
        if i not in opened and i != "OoB" and (not guessed or i not in stated_safe):
            local_unkws += 1
    return local_unkws
#%%
model = Sequential([
    Input(shape=(7,7,1)),
    Conv2D(64, (3, 3), activation='relu',),
    Dropout(0.2),

    Conv2D(256, (3, 3), activation='relu',),
    Flatten(),

    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    Dense(2, activation='softmax')
])
#     |||    7.0    |||
tf.keras.mixed_precision.set_global_policy('mixed_float16')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 1.0 - Original
# 2.0 - Simulated layouts added
# 2.5 - Added augmentations (rotation, mirroring layouts), new model
# 3.0 - Removed skip, added flags recheck and wipe [failed], new model
# 4.0 - Brand-new layout format
# 4.1 - Taught to count to 8, added anti-bomb-overflowing
# 4.2 - Double-check before clicking in 50-50 situations   - FIRST SOLVE, but rare
# 4.3 - fixed random bug that ruined a lot (picking not the best known tile)  - more solves
# 4.4 - corner priority - more solves
# 5.0 - 7x7 vision data format.
# 5.1 - slightly different model
# 6.0 - auto-brain.
# 6.1 - extended model.
# 7.0 - underflow prevention. Final version. Lose pretty much only in 50/50 situations (have at least 1 map layout,
# when selected tile is a mine, and at least 1, when selected tile is safe)
# End.
#%%
# model = load_model("ext_model.keras")

# DataRaw = [i for i in simulate_data(1000,1)]
# DataAuged = [j for i in DataRaw for j in all_iso_variants(i)]
# X = np.array([i[0] for i in DataAuged])
# y = [i[1]-1 for i in DataAuged]
#
# y_one_hot = tf.keras.utils.to_categorical(y, num_classes=2)
#
# model.fit(X, y_one_hot, epochs=10, batch_size=128, validation_split=0.2)

# model.save("ext_model.keras")
#%%
model = load_model("ext_model.keras")

map_width = 15
mine_freq = 0.17
too_fast = True
while too_fast:
    already_lost = False
    too_fast = True
    dead_end = False
    disable_recheck = False
    opened = []
    knowledge = []
    flags = []
    flags_chances = []
    temporal_uncertainty = []
    iteration = 0
    prev_pick = ()
    pick = ()
    decision = 0
    map = generate_map(map_width,mine_freq)
    mines_amount = sum([1 if i == 9 else 0 for i in map.flatten().tolist()])
    print_map(map, opened)

    pick = (np.random.randint(1,map_width-2),np.random.randint(1,map_width-2))
    while map[pick] == 9:
        pick = (np.random.randint(1,map_width-2),np.random.randint(1,map_width-2))
    print(pick)
    recursive_open(pick)
    empty_recursive_cache()
    print_map(map, opened)


    while len(opened) < map_width**2 and not already_lost:
        flags_chances = []
        if len(all_available()) <= len(temporal_uncertainty):
            temporal_uncertainty = []
        print("-"*map_width*3)
        print_map(map, opened)
        iteration += 1

        if len(opened) > 8:
            too_fast = False


        stated_safe = []
        for i in range(8):
            for only_flag in set([i for j in all_available() for i in get_adjacent(j) if i in opened and count_unknowns(i) == map[i]-count_known_mines(i)]):
                for tile in get_adjacent(only_flag):
                    if tile not in opened and tile != "OoB":
                        print(f"auto-flag {tile}")
                        # print(tile)
                        # print("FROM")
                        # print(only_flag)
                        flags.append(tile)
                        opened.append(tile)

            for clear in set([i for j in all_available() for i in get_adjacent(j) if i in opened and count_known_mines(i) == map[i]]):
                for tile in get_adjacent(clear):
                    if tile not in opened and tile != "OoB":
                        if tile not in stated_safe:
                            stated_safe.append(tile)
                            print(f"consider safe {tile}")

            #check for underflow
            if decision == 2:
                prev_pick = pick
            for tile in stated_safe:
                for risk in get_adjacent(tile):
                    if risk != "OoB" and risk in opened and risk not in flags:
                        if count_unknowns(risk, True) < map[risk] - count_known_mines(risk):
                            print(f"Local underflow {risk}")
                            if prev_pick != ():
                                opened.remove(prev_pick)
                                flags.remove(prev_pick)
                                print("Prev. flag prooved WRONG. Revert")
                                if map[prev_pick] == 9:
                                    print(cr.Fore.RED + "<< LOSS >> (?)" + cr.Fore.RESET)
                                    already_lost = True
                                else:
                                    print("Successful click (Reverted flag)")
                                    recursive_open(prev_pick)
                                    empty_recursive_cache()
                                stated_safe = []

        for clear in set([i for j in all_available() for i in get_adjacent(j) if i in opened and count_known_mines(i) == map[i]]):
            for tile in get_adjacent(clear):
                if tile not in opened and tile != "OoB":
                    print(f"auto-clear {tile}")
                    if map[tile] == 9:
                        print(cr.Fore.RED + "<< LOSS >>" + cr.Fore.RESET)
                        already_lost = True
                    else:
                        print("Successful click")
                        recursive_open(tile)
                        empty_recursive_cache()
        print("-"*map_width*3)
        print_map(map, opened)



        if len(opened) >= map_width**2 or already_lost:
            break


        if not_flags() > 3:
            knowledges = get_most_known()
            pick = tuple([int(i) for i in knowledges[np.array([float(i) for i in knowledges[:, 1]]).argsort()[-1], 0].split(",")])
            # print(knowledges)
            # print([int(i) for i in knowledges[:, 1]])
            # print(np.array([int(i) for i in knowledges[:, 1]]).argsort())
            ladder_count = 0
            while pick in temporal_uncertainty:
                ladder_count += 1
                if ladder_count > len(knowledges):
                    print("i think this is a dead end")
                    if disable_recheck:
                        dead_end = True
                    disable_recheck = True
                    temporal_uncertainty = []
                else:
                    pick = tuple([int(i) for i in knowledges[np.array([float(i) for i in knowledges[:, 1]]).argsort()[-1-ladder_count], 0].split(",")])
        else:
            pick = list(all_available())[np.random.randint(0,len(all_available()))]
            while pick in temporal_uncertainty:
                pick = list(all_available())[np.random.randint(0,len(all_available()))]
        if dead_end:
            pick = (np.random.randint(0,map_width-1),np.random.randint(0,map_width-1))
            while pick in temporal_uncertainty or pick in opened:
                pick = (np.random.randint(0,map_width-1),np.random.randint(0,map_width-1))
        print(pick)
        predictions = model.predict(np.array([get_slice(pick)]))
        print(predictions)
        predicted_classes = np.argmax(predictions, axis=1) + 1
        decision = predicted_classes[0]
        print(decision)

        if decision == 1 or not_flags() < 4:
            if pick not in temporal_uncertainty:
                if map[pick] == 9:
                    print(cr.Fore.RED + "<< LOSS >>" + cr.Fore.RESET)
                    already_lost = True
                else:
                    print("Successful click")
                    recursive_open(pick)
                    empty_recursive_cache()
                    local_mines = []
                    for i in get_adjacent(pick):
                        if i in flags:
                            local_mines.append(i)
                    if len(local_mines) > map[pick] and not disable_recheck:
                        print("Local overflow!")
                        for i in range(len(local_mines) - int(map[pick])):
                            mine_chances = []
                            for m in local_mines:
                                print(m)
                                predictions = model.predict(np.array([get_slice(m, True)]))
                                print(predictions)
                                predicted_classes = np.argmax(predictions, axis=1) + 1
                                mine_chances.append(predictions[0][1])
                            low = local_mines[mine_chances.index(min(mine_chances))]
                            print("Removed:")
                            print(low)
                            flags.remove(low)
                            opened.remove(low)
                            temporal_uncertainty.append(low)
        elif decision == 2 and pick not in temporal_uncertainty:
            print("< Flag >")
            flags.append(pick)
            opened.append(pick)
            for tile in [i for i in get_adjacent(pick) if i not in flags and i in opened]:
                local_mines = []
                for i in get_adjacent(tile):
                    if i in flags:
                        local_mines.append(i)
                if len(local_mines) > map[tile] and not disable_recheck:
                    print("Local overflow!")
                    for i in range(len(local_mines) - int(map[tile])):
                        mine_chances = []
                        for m in local_mines:
                            print(m)
                            predictions = model.predict(np.array([get_slice(m, True)]))
                            print(predictions)
                            predicted_classes = np.argmax(predictions, axis=1) + 1
                            mine_chances.append(predictions[0][1])
                        low = local_mines[mine_chances.index(min(mine_chances))]
                        print("Removed:")
                        print(low)
                        flags.remove(low)
                        opened.remove(low)
                        temporal_uncertainty.append(low)

if len(opened) >= map_width**2 and not already_lost:
    print(cr.Fore.GREEN + "<< SUCCESS >>")
elif already_lost:
    print(cr.Fore.RED + "<< LOSS >>" + cr.Fore.RESET)
