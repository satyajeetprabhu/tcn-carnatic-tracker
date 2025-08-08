#import madmom
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, DBNBarTrackingProcessor
from scipy.ndimage import maximum_filter1d
import numpy as np

fps= 100  # frames per second for the DBN processors

def beat_tracker(beats_act, downbeats_act):
    beat_dbn = DBNBeatTrackingProcessor(
        min_bpm=55.0, max_bpm=215.0, fps=fps, transition_lambda=100, online=False)
            
    if beats_act.size > 1:
        beats_pred = beat_dbn(beats_act)
        return beats_pred, np.array([])  # No downbeats in this tracker
    else:
        # If no beats are detected, return an empty array
        return np.array([]), np.array([])

def joint_tracker(beats_act, downbeats_act):
    
    downbeat_tracker = DBNDownBeatTrackingProcessor(
                        beats_per_bar=[3, 5, 7, 8], min_bpm=55.0, max_bpm=300.0, fps=100)

    
    combined_act = np.vstack((np.maximum(beats_act - downbeats_act, 0), downbeats_act)).T
    pred = downbeat_tracker(combined_act)
    
    downbeats_pred = pred[pred[:, 1] == 1][:, 0]
    beats_pred = pred[:, 0]
    
    return beats_pred, downbeats_pred

def sequential_tracker(beats_act, downbeats_act):
    
    beats, _ = beat_tracker(beats_act, downbeats_act)
    
    # bars (i.e. track beats and then downbeats)
    beat_idx = (beats * fps).astype(int)
    bar_act = maximum_filter1d(downbeats_act, size=3)
    bar_act = bar_act[beat_idx]
    bar_act = np.vstack((beats, bar_act)).T
    
    bar_tracker = DBNBarTrackingProcessor(beats_per_bar=(3, 5, 7, 8), meter_change_prob=1e-3, observation_weight=4)
    
    try:
        pred = bar_tracker(bar_act)
    except IndexError:
        pred = np.empty((0, 2))
    
    downbeats_pred = pred[pred[:, 1] == 1][:, 0]
    beats_pred = pred[:, 0]
    
    return beats_pred, downbeats_pred