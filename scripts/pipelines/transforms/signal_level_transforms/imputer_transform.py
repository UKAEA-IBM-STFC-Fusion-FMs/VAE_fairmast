import numpy as np
import os
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# RuntimeWarning are not printed to terminal. 
# They follow the np.mean operation when the array is all NaN.
# These special case are properly dealt with by applying SimpleImputer

cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(mother_dir)


class ImputerTransform(object):
    """
    Replace NaN entries with channel mean value obtained from non-NaN entries.
    
    For channels that feature only NaN entries, these are imputed 
    column-wise using non-NaN entries from other channels.

    Finally, the signal (all channels) are standardized.
    
    This imputer was developed for the VAE latent space representation.
    
    Parameters:
    sample: dict {"values":vals, "time":time}
    
    Return:
    signal after imputation of NaN entires.
    """
    def __init__(self):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        
    def __call__(self, sample):
        
        # Retireve "values" and "time" from the sample
        try:
            vals, time = sample["values"], sample["time"]
        except KeyError as e:
            print(f"KeyError: {e}. Sample is missing required keys.")
            return None

        if vals is None or time is None:
            return None
        
        
        if vals.ndim == 2:
            
            # Compute means along rows, ignoring NaNs
            row_means = np.nanmean(vals, axis=1)
            
            # Replace NaN with corresponding mean
            for i in range(vals.shape[0]):
                ith_channel = vals[i]
                ith_channel[np.isnan(ith_channel)] = row_means[i]
                vals[i] = ith_channel
                
            # Check for empty channels
            if np.any(np.isnan(row_means)):
                vals = self.imputer.fit_transform(vals)
                
        elif vals.ndim == 1:
            if np.any(np.isnan(vals)):
                mean_val = np.nanmean(vals)
                if np.isnan(mean_val):  # means all values were NaN
                    return None
                else:
                    vals[np.isnan(vals)] = mean_val          
        else:
            print("Error in imputer_transform.py, vals dimension must be 1 or 2.")
            return None
                             
        return {"values":vals, "time":time}
    
    
def test_imputer_tranform():
    imputer = ImputerTransform()
    
    v = np.array([
        [5,3,2],
        [1.0, np.nan, 2.0],
        [np.nan, np.nan, np.nan],
        [10,6,4]])

    t = np.array([
        [ 1,2,3],
        [1,2,3],
        [1,2,3],
        [1,2,3]])
    
    
    print(f" Before imputer {v}")
    sample = imputer({"values" : v, "time":t})
    if sample is not None:
        print(f" After imputer {sample['values']}")
    else:
        print(f" After imputer {sample}")
    
    v =  np.array([0,2,3,4,np.nan])
    print(f" Before imputer {v}")
    sample = imputer({"values" : v, "time":t})
    if sample is not None:
        print(f" After imputer {sample['values']}")
    else:
        print(f" After imputer {sample}")
    
    v =  np.array([np.nan, np.nan, np.nan])
    print(f" Before imputer {v}")
    sample = imputer({"values" : v, "time":t})
    if sample is not None:
        print(f" After imputer {sample['values']}")
    else:
        print(f" After imputer {sample}")

if __name__ == "__main__":
    test_imputer_tranform()
    