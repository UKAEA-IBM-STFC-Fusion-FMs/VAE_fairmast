import pickle
import numpy as np 

mean_path = "../../metadata/2025-09-22_stdscaling/dict_mean_shot.pkl"
std_path = "../../metadata/2025-09-22_stdscaling/dict_std_shot.pkl"

with open(mean_path, "rb") as f:  # 'rb' = read binary
    dict_mean = pickle.load(f)
with open(std_path, "rb") as f:  # 'rb' = read binary
    dict_std = pickle.load(f)

flattened_dict_mean = dict_mean
for var in flattened_dict_mean.keys():
    # print( flattened_dict_mean[var].shape )
    mean_value = np.nanmean( dict_mean[var] ) 
    mean_arr = np.full_like(dict_mean[var], mean_value)
    flattened_dict_mean[var] = mean_arr

flattened_dict_std = dict_std
for var in flattened_dict_std.keys():
    # print( flattened_dict_std[var].shape )
    std_value = np.nanmean( dict_std[var] ) 
    std_arr = np.full_like(dict_std[var], std_value)
    flattened_dict_std[var] = std_arr

with open('preprocessing/flattened_dict_mean_shot.pkl', 'wb') as f_:
        pickle.dump(flattened_dict_mean, f_)
with open('preprocessing/flattened_dict_std_shot.pkl', 'wb') as f_:
    pickle.dump(flattened_dict_std, f_)