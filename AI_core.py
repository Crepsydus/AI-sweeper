import numpy as np
import colorama as cr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
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
            # if adj == "OoB":
            #     knowledge += 1
            if adj in opened:
                if adj not in flags:
                    knowledge += 2
                else:
                    knowledge += 1
        knowledge += 2 * count_open_corner(coords)
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
                    if i in flags and not hide_flags:
                        local_km += 1
                    if (i not in opened and i != "OoB") or (i in flags and hide_flags):
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
def all_iso_variants(array):
    nums_9 = array[0][0:8]
    nums_9.insert(4,99)
    nums_9 = np.array(nums_9).reshape(3,3)

    kms_9 = array[0][8:16]
    kms_9.insert(4,99)
    kms_9 = np.array(kms_9).reshape(3,3)

    unkws_9 = array[0][16:24]
    unkws_9.insert(4,99)
    unkws_9 = np.array(unkws_9).reshape(3,3)

    nums_1 = np.rot90(nums_9)
    kms_1 = np.rot90(kms_9)
    unkws_1 = np.rot90(unkws_9)

    nums_2 = np.rot90(nums_9,2)
    kms_2 = np.rot90(kms_9, 2)
    unkws_2 = np.rot90(unkws_9, 2)

    nums_3 = np.rot90(nums_9,3)
    kms_3 = np.rot90(kms_9,3)
    unkws_3 = np.rot90(unkws_9,3)

    nums_9f = np.fliplr(nums_9)
    kms_9f = np.fliplr(kms_9)
    unkws_9f = np.fliplr(unkws_9)
    nums_1f = np.fliplr(nums_1)
    kms_1f = np.fliplr(kms_1)
    unkws_1f = np.fliplr(unkws_1)
    nums_2f = np.fliplr(nums_2)
    kms_2f = np.fliplr(kms_2)
    unkws_2f = np.fliplr(unkws_2)
    nums_3f = np.fliplr(nums_3)
    kms_3f = np.fliplr(kms_3)
    unkws_3f = np.fliplr(unkws_3)

    count = 0
    result = []
    for i in [nums_9,kms_9,unkws_9,nums_1,kms_1,unkws_1,nums_2,kms_2,unkws_2,nums_3,kms_3,unkws_3,nums_9f,kms_9f,unkws_9f,
              nums_1f,kms_1f,unkws_1f,nums_2f,kms_2f,unkws_2f,nums_3f,kms_3f,unkws_3f]:
        count += 1
        ii = i.flatten().tolist()
        ii.pop(4)
        result = result + ii
        if count == 3:
            count = 0
            yield (result, array[1])
            result = []
def not_flags():
    return sum([1 if i not in flags else 0 for i in opened])

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
#%%
model = Sequential([
    Input(shape=(24,)),
    Dense(128, activation='relu'),  # Входной слой с 128 нейронами
    BatchNormalization(),                              # Нормализация для ускорения сходимости
    Dropout(0.3),                                      # Dropout для предотвращения переобучения

    Dense(256, activation='relu'),                     # Скрытый слой с 256 нейронами
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),                     # Ещё один скрытый слой
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),                      # Ещё один скрытый слой
    Dense(2, activation='softmax')                     # Выходной слой для 3 классов
])
#     |||    4.3    |||

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

# 4.4 - corner priority
#%%
DataRaw = [i for i in simulate_data(50000,0.33)]
DataAuged = [j for i in DataRaw for j in all_iso_variants(i)]
X = np.array([i[0] for i in DataAuged])
y = [i[1]-1 for i in DataAuged]

y_one_hot = tf.keras.utils.to_categorical(y, num_classes=2)

model.fit(X, y_one_hot, epochs=10, batch_size=50, validation_split=0.2)

#%%
# DataRaw = [i for i in simulate_data(1,0.1)]
# print([i for i in all_iso_variants(DataRaw[0])])
#%%
#inputs
map_width = 7
mine_freq = 0.17

#internal variables
opened = []
knowledge = []
flags = []
flags_chances = []
temporal_uncertainty = []
iteration = 0
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


while len(opened) < map_width**2:
    dead_end = False
    flags_chances = []
    if len(all_available()) <= len(temporal_uncertainty):
        temporal_uncertainty = []
    print("-"*map_width*2)
    print_map(map, opened)
    iteration += 1

    # for tile in opened:
    #     if tile not in flags and map[tile] != 0:
    #         local_unkws = 0
    #         local_mines = 0
    #         ready_to_open = []
    #         for adj in get_adjacent(tile):
    #             if adj not in opened  and adj != "OoB":
    #                 local_unkws += 1
    #                 ready_to_open.append(adj)
    #             if adj in flags:
    #                 local_mines += 1
    #                 local_unkws += 1
    #         if local_unkws == map[tile] and local_mines != local_unkws:
    #             for i in ready_to_open:
    #                 flags.append(i)
    #                 opened.append(i)

    if not_flags() > 3:
        knowledges = get_most_known()
        pick = tuple([int(i) for i in knowledges[np.array([int(i) for i in knowledges[:, 1]]).argsort()[-1], 0].split(",")])
        # print(knowledges)
        # print([int(i) for i in knowledges[:, 1]])
        # print(np.array([int(i) for i in knowledges[:, 1]]).argsort())
        ladder_count = 0
        while pick in temporal_uncertainty:
            ladder_count += 1
            if ladder_count > len(knowledges):
                print("i think this is a dead end")
                dead_end = True
                break
            else:
                pick = tuple([int(i) for i in knowledges[knowledges[:, 1].argsort()][-1-ladder_count, 0].split(",")])
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
        # blind_predictions = model.predict(np.array([get_slice(pick, True)]))
        # blind_classes = np.argmax(blind_predictions, axis=1) + 1
        # if blind_classes[0] != decision and blind_predictions[0][1]-predictions[0][1] > 0.3:
        #     print("Vague place")
        #     temporal_uncertainty.append(pick)
        #     for tile in get_adjacent(pick):
        #         if tile in flags:
        #             flags.remove(tile)
        #             opened.remove(tile)
        #             temporal_uncertainty.append(tile)
        if pick not in temporal_uncertainty:
            if map[pick] == 9:
                print(cr.Fore.RED + "<< LOSS >>" + cr.Fore.RESET)
                break
            else:
                print("Successful click")
                recursive_open(pick)
                empty_recursive_cache()
    elif decision == 2 and pick not in temporal_uncertainty:
        # if len(flags) <= mines_amount:
        print("< Flag >")
        flags.append(pick)
        opened.append(pick)
        for tile in [i for i in get_adjacent(pick) if i not in flags and i in opened]:
            local_mines = []
            for i in get_adjacent(tile):
                if i in flags:
                    local_mines.append(i)
            if len(local_mines) > map[tile]:
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
