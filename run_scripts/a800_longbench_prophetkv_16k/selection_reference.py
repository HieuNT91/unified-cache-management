"""Independent Python ranking oracle for original ProphetKV."""
import math


def native_reference(scores, prefix, parameters):
    ratio = parameters['total_ratio']
    if not 0 <= ratio <= 1 or any(not math.isfinite(x) for x in scores):
        raise ValueError('Invalid scores or ratio')
    count = math.floor(len(scores) * ratio)
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return sorted(prefix+i for i in ranked[:count]), {}
