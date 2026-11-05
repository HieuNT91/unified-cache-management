"""Explicit TP contract shared by collection, replay and resource checks."""
def validate_tp(tp):
    if type(tp) is not int or tp not in (2, 4):
        raise ValueError('Qwen3 collection supports TP2 or TP4')
    return tp


def protocol_tp(protocol):
    tp = validate_tp(protocol.get('tp', 4))
    if any(len(group) != tp for group in protocol['groups']):
        raise ValueError('GPU group size differs from tensor parallel size')
    return tp
