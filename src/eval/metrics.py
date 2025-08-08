import mir_eval

def all_metrics(gt_times, pred_times):
    
    reference = mir_eval.beat.trim_beats(gt_times)
    estimated = mir_eval.beat.trim_beats(pred_times)
    
    # Compute the beat evaluation metrics
    scores = mir_eval.beat.evaluate(reference, estimated)
    
    return scores

def flatten_dict(track_id, beat_scores, downbeat_scores):
    flat_result = {'track_id': track_id}
    
    for k, v in beat_scores.items():
        flat_result[f'beat_{k}'] = v
    for k, v in downbeat_scores.items():
        flat_result[f'downbeat_{k}'] = v
    
    return flat_result