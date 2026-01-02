import pandas as pd


# ======================================================================================================================
class FillThomsonWithZerosTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_):
        """
        Input: torch dict with key 'time' and key 'values'.

        Returns: torch dict with key 'time' and key 'values with NaNs of profiles (i.e., when one full channel in the
        profile is missing) filled with zeros.
        """

        time = dict_['time']
        values = dict_['values']
        df = pd.DataFrame(values)
        # print('\nBefore filling with zeros: ', list(df.isna().sum(axis=0)))
        for col in df.columns:
            if not df[col].isna().all():
                df[col] = df[col].fillna(value=0)
        # print('After filling with zeros: ', list(df.isna().sum(axis=0)))
        
        return {
            'time': time,
            'values': df.values
        }

    # ------------------------------------------------------------------------------------------------------------------
