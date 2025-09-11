
from collections import defaultdict
import torch

class SegmenterTransform(object):
    '''
     Parameters
    ----------
    time_window_sec : float
        The length of the time window in seconds to segment the x and y values.
    stride_sec : float
        The step in seconds to move the time window.
    offset : float, optional
        The offset in seconds to start the time window from the end of the signal, by default 0.0.
        This is also used to offset the y segments in time with respect to the x segments.
    '''
    def __init__(
        self,  
        time_window_sec, 
        stride_sec, 
        offset
        ):
        
        
        self.time_intervals_checks= False
        if not (
                time_window_sec > 0 and \
                stride_sec > 0 and \
                offset >= 0 and \
                stride_sec < time_window_sec 
            ):
            self.time_intervals_checks = True
            raise ValueError(
                "Invalid parameters for time window, time step or offset. " +
                "Ensure that time_window_sec > 0, stride_sec > 0, offset >= 0, " +
                "stride_sec < time_window_sec and offset <= time_window_sec. \n" +
                f" Current time_window_sec {time_window_sec}\n"+
                f" Current offset {offset}\n"+
                f" Current stride_sec {stride_sec}\n"
            )
        
        self.time_window_sec = time_window_sec
        self.stride_sec =  stride_sec
        self.offset =  offset

    def __call__(self, shot):    
        return segment_shot(
                shot,  
                self.time_window_sec, 
                self.stride_sec, 
                self.offset
            )

    
def segment_data_in_time_windows(
    values,
    times,
    time_window, 
    stride_sec,
    offset=0.0
    ):
    """ Segment the "values" and "times" into smaller segments 
        each containing the original data in a given time window.
    Parameters
    ----------
    values : np.ndarray
        The values to segment, shape: (n_features, time_steps).
    times : np.ndarray
        The times corresponding to the values, shape: (time_steps,).
    time_window : float
        The length of the time window in seconds.
    stride_sec : float
        The step in seconds to move the time window.
    offset : float, optional
        The offset in seconds to start the time window from the end of the signal, by default 0.0.
        This is also used to offset the y segments in time with respect to the x segments when we
        what to to forecasting of y w.r.t. x.
    Returns
    -------
        Two lists one for "values" and one for "times" containing segments.
    """

    values = torch.tensor(values,dtype=torch.float32)  # shape: (n_features, time_steps)
    times = torch.from_numpy(times)    # shape: (time_steps,)

    start_time = times[-1].item() - time_window - offset
    segments_v = []
    segments_t = []

    while start_time >= times[0].item():
        
        # Find the indeces where the times is in the time window
        mask = (times >= start_time) & (times < start_time + time_window)
                
        # Return list of True indeces
        idx = mask.nonzero(as_tuple=True)[0]
        
        # If no end index is found, take the whole time series
        if len(idx) == 0:
            print(f"No time values found in {start_time}sec - {start_time + time_window}sec")
            segments_v.insert(0,values)
            segments_t.insert(0,times)
            break

        val_segment =  values[:, idx[0]:idx[-1] + 1]
        time_segment = times[idx[0]:idx[-1] + 1]

        segments_v.insert(0,val_segment)
        segments_t.insert(0,time_segment)

        start_time -=  stride_sec

    return segments_v, segments_t


def create_map(shot):
    """Create indexed map that contains mini-dictionaries, 
        one per segment of values and times 
        for all the signals of found in the shot"""
    # Create a defaultdict to hold the segments
    data_map = defaultdict(list)

    # For each signal in the base dictionary
    for source_signal_names, sample in shot.items():
        try:
            values = sample["values"] # list of segments
            times = sample["time"] # list of segments
        except KeyError as e:
            print(f"KeyError: {e}. Sample is missing required keys.")
            return None
        
        
        # segment_data_in_time_windows return a list of time segments of fixed length.
        # However, the last segment may be shorter than the others is the signal is not long enough
        # to contain a full time window.
        # Therefore, we will always remove the last segment independentkly of its length.
        for idx, (values_segment, time_segment) in enumerate(zip(values, times)):
            if idx == len(times) - 1:
                continue    # Skip last segment, since this might be shorter than the others, see explanation above
            
            data_map[idx].append({
                "signal_name": source_signal_names, 
                "values": values_segment, 
                "time": time_segment
            })
            
    return data_map

def segment_shot(
    shot,  
    time_window_sec, 
    stride_sec, 
    offset):
    """Segment the dictionaries contained in sample into smaller 
       dictionaries each containing a segment of the original arrays.
       
       These segments are created by sliding a time window of length
       time_window_sec over array values with a step of stride_sec. This algorithm is
       implemented in the function segment_data_in_time_windows.
       
       HINT: for forecasting purposes, the segments of y are 
       forward in time with respect to the segments of x. 
    
    Parameters
    ----------
    shot : dict
        An item from the MastDataset single shot level transform
    time_window_sec : int
        The length of the time window in seconds to segment the x and y values.
    stride_sec : int
        The step in seconds to move the time window.
    offset : int
        The offset in seconds to start the time window from the end of the signal.
    Returns
    -------
        A list of sub-dictionaries each one containing a time segment 
        of the original dictionary, for all the signals in the original shot. 
        For instance, the first element of the list will contain:
        {       
                "shot_id": shot_id
                "source_name-signal_name": [
                       {
                           "signal_name": signal["names"], 
                            "values": values_segment, 
                            "time": time_segment
                        },
                        {
                          ...
                        },
                        ...
                    ]
        }
                
        all structures { ... } correspond to the first time segment
        of the original x.
    """
    if not shot:
        return None
    if type(shot) is not dict:
        raise TypeError(f"Shot must be a dictionary and not {type(shot)}.")
   
    for _, sample in shot.items():
        try:
            values, times = sample["values"], sample["time"]
        except KeyError as e:
            print(f"KeyError: {e}. Sample is missing required keys.")
            return None

        # Segmet "values" and "times" in x and y
        try:
            segments_v, segments_t = segment_data_in_time_windows(
                values,
                times, 
                time_window_sec, 
                stride_sec, 
                offset
            )
            sample["values"] = segments_v
            sample["time"] = segments_t
        except AssertionError as e:
            print(f"AssertionError: {e}")
            return None
        
    # Create map of segments
    x_map = create_map(shot)
              
    # Create a list of new mini dictionaries x, one for each temporal segment
    x_list = []
    for idx in range(len(x_map)):
        x_mini_dict = {
            "sources_signals": x_map[idx]
        }
        x_list.append(x_mini_dict)
        
    return x_list


if __name__ == "__main__":
    
    train_shots = [30207]
    data_names =[["magnetics","flux_loop_flux"]]
    target_names =[["magnetics", "b_field_tor_probe_saddle_voltage"]]

    import os
    import sys
    
    sys.path.append("scripts/MAST_tools")
    from MAST_dataset import MastDataset
    from  signal_utils import MASTSignalManager
    import numpy as np
    
    sig = MASTSignalManager()
    store_manager = sig.store_manager
    store = store_manager.make_shot_store(
        shot_info={
                "shot_id": train_shots[0],
                "local": False  # Add among attributes
                }
        )
          
    data_profile = sig.get_signal_profile(
                    data_origin=store,
                    source_name=data_names[0][0],
                    signal_name=data_names[0][1]
                    )  
    target_profile = sig.get_signal_profile(
                    data_origin=store,
                    source_name=target_names[0][0],
                    signal_name=target_names[0][1]
                    )  
    
    def make_shot_dictionary(shot_profile, signal, store, source):
       
        try:
            shot_time, _ = sig.get_signal_times_and_time_type(
                                signal,
                                store,
                                source
                                )
        except Exception as e:
            print("Error getting time for shot")
            shot_time = None
        
        try:
            shot_vals = (np.expand_dims(shot_profile.values, axis=0) if shot_profile.values.ndim == 1
                                 else shot_profile.values)
        except AttributeError:
            shot_vals = None
            
            
        # Apply variable-level transforms
        shot = {}
        shot[f'{source}-{signal}'] = {"time": shot_time, "values": shot_vals, "source-signal": f"{source}-{signal}"}
        
        return shot

    data_shot = make_shot_dictionary(data_profile, data_names[0][1], store, data_names[0][0])
    target_shot = make_shot_dictionary(target_profile, target_names[0][1], store, target_names[0][0])
    
    time_window_sec =  0.01
    stride_sec = 0.005
    offset = 0.003
    

    # Working
    list_x = segment_shot(data_shot, time_window_sec, stride_sec, offset)
    list_y = segment_shot(target_shot, offset, stride_sec, offset=00)
       
    
    # Printing the first two segments of x and y
    x0=list_x[-1]
    y0=list_y[-1]
    
    x1=list_x[-2]
    y1=list_y[-2]
    
    print(f"x0 times: {x0['sources_signals'][0]['time']}")
    print(f"x1 times: {x1['sources_signals'][0]['time']}")
    print(f"y0 times: {y0['sources_signals'][0]['time']}")
    print(f"y1 times: {y1['sources_signals'][0]['time']}")