"""Explicit typed byte packets for vLLM's untyped collective-RPC responses."""
import numpy as np


def pack(value):
    if isinstance(value,np.ndarray):
        if value.dtype!=np.float32:raise ValueError('Study transfer requires FP32 arrays')
        return {'__study_fp32__':True,'shape':list(value.shape),'data':value.tobytes(order='C')}
    if isinstance(value,dict):return {k:pack(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [pack(v) for v in value]
    return value


def unpack(value):
    if isinstance(value,dict):
        if '__study_fp32__' in value:
            if set(value)!={'__study_fp32__','shape','data'} or value['__study_fp32__'] is not True:raise ValueError('Malformed array transfer')
            shape=value['shape']
            if not isinstance(shape,list) or not shape or any(type(n) is not int or n<0 for n in shape):raise ValueError('Malformed array dimensions')
            size=1
            for n in shape:size*=n
            if not isinstance(value['data'],bytes) or len(value['data'])!=size*4:raise ValueError('Truncated FP32 transfer')
            return np.frombuffer(value['data'],dtype=np.float32).reshape(shape)
        return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [unpack(v) for v in value]
    return value
