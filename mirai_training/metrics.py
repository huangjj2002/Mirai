"""Annual AUC plus explicitly distinguished legacy and censoring-weighted concordance."""
import numpy as np
from sklearn.metrics import roc_auc_score


def km_grid(times, event):
    times, event = np.asarray(times), np.asarray(event, dtype=bool)
    before, after, survival = [], [], 1.0
    for t in range(5):
        before.append(survival)
        n = int((times >= t).sum())
        d = int(((times == t) & event).sum())
        if n:
            survival *= 1-d/n
        after.append(survival)
    return np.array(before), np.array(after)


def concordance(times, events, probs, train_times, train_events, legacy=False):
    """O(5 n log n); year-bin convention treats an event/censor tie as comparable.

    legacy=True reproduces upstream horizon-at-later-exit and cancer-survival weighting.
    legacy=False uses event-horizon scores and train-set censoring KM at t-.
    Neither result claims recovery of continuous event times from binned metadata.
    """
    times, events, probs = np.asarray(times), np.asarray(events,dtype=bool), np.asarray(probs)
    g = km_grid(train_times, train_events if legacy else np.logical_not(train_events))[1 if legacy else 0]
    correct = pairs = 0.0
    for t in range(5):
        if legacy:
            previous = probs[(times < t) & events, t]
            current = probs[(times == t) & events, t]
            censored = probs[(times == t) & ~events, t]
            batches = [(previous,current), (np.concatenate([previous,current]),censored)]
        else:
            # Earlier events should have greater risk than comparable later/censored exams.
            previous = probs[(times == t) & events,t]
            current = probs[(times > t) | ((times == t) & ~events),t]
            batches = [(previous,current)]
        for high,low in batches:
            if not len(high) or not len(low):
                continue
            if g[t] <= 0:
                return None
            sorted_high = np.sort(high)
            lt = np.searchsorted(sorted_high,low,side='left')
            le = np.searchsorted(sorted_high,low,side='right')
            correct += (len(high)-le + .5*(le-lt)).sum()/g[t]**2
            pairs += len(high)*len(low)/g[t]**2
    return float(correct/pairs) if pairs else None


def evaluate_metrics(times, events, probs, train_times, train_events):
    t,e,p = np.asarray(times),np.asarray(events,dtype=bool),np.asarray(probs)
    result = {}
    for j in range(5):
        positive = e & (t <= j)
        include = positive | (t >= j)
        y = positive[include].astype(int)
        result[f'{j+1}_year_auc'] = float(roc_auc_score(y,p[include,j])) if len(np.unique(y))==2 else None
        result[f'{j+1}_year_n'] = int(include.sum())
        result[f'{j+1}_year_events'] = int(y.sum())
    result['mirai_legacy_c_index'] = concordance(t,e,p,train_times,train_events,legacy=True)
    result['uno_discrete_c_index'] = concordance(t,e,p,train_times,train_events,legacy=False)
    return result
