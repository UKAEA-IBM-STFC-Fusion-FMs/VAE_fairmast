import joblib
import numpy as np

class PCATransform(object):
    """Use a pre-fitted PCA function to transform input data.

    Parameters
    ----------
    pca_model_path : str
        Path to fitted pca model joblib file.
    """

    def __init__(self, model_pca):
        self.model_pca = model_pca

    def __call__(self, sample):
        try:
            vals, time = sample["values"], sample["time"]
        except KeyError as e:
            print(f"KeyError: {e}. Sample is missing required keys.")
            return None

        if vals is None or len(vals) == 0:
            return None
        if time is None or len(time) == 0:
            return None
        
        # Select model type and signal from those available in the dictionary.
        pca_model =  self.model_pca
                               
        # Apply Scaler and PCA
        if not np.isnan(vals).any():
            
            #Import models
            pca = pca_model["pca"]
            scaler = pca_model["scaler"]
            
            # Transoform data in PCA space
            try:
                x_scaled = scaler.transform(vals.T) 
                x_transform = pca.transform(x_scaled)
                #vals = scaler.inverse_transform(x_transform).T
                vals = x_transform.T
            except Exception as e:
                print(f"[ERROR] PCA transformation failed for signal '{source_signal}' Exception: {e}")
                return None
            
            return {"values":vals, "time":time}
        else:
            return None
