import pandas as pd


# ======================================================================================================================
class FillWithZerosImputerTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_):
        """
        Input: torch dict with key 'time' and key 'values'
        Returns: torch dict with key 'time' and key 'values with NaNs filled with zeros
        """

        time = dict_['time']
        values = dict_['values']
        df = pd.DataFrame(values.T)
        # print('\nBefore filling with zeros: ', list(df.isna().sum(axis=0)))
        df = df.fillna(value=0)
        # print('After filling with zeros: ', list(df.isna().sum(axis=0)))
        
        return {
            'time': time,
            'values': df.values.T
        }

    # ------------------------------------------------------------------------------------------------------------------
