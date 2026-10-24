"""Tree-only collection identity; separate from router action inventories."""
from runner.tree_prophetkv import METHOD, PROTOCOL

SCHEMA = 'tree-prophetkv-thinking-collection-v1'
RATIOS = (5, 20, 40, 60)


def actions(percentages):
    if (not percentages or len(set(percentages)) != len(percentages)
            or any(type(p) is not int or not 0 < p <= 100 for p in percentages)):
        raise ValueError('Expected distinct integer percentages in [1, 100]')
    return [dict(id=f'{METHOD}-{p}', method=METHOD, ratio=p/100) for p in percentages]


def inventory(protocol):
    expected = actions(protocol['percentages'])
    if (protocol.get('schema') != SCHEMA or protocol.get('tree_protocol') != PROTOCOL
            or protocol.get('actions') != expected
            or protocol.get('scheduled_actions') != [a['id'] for a in expected]
            or protocol.get('phase_order') != ['cached']
            or protocol.get('answer_validation') != 'native-answer-diagnostics-v1'):
        raise ValueError('Invalid tree-prophetkv collection protocol')
    return {a['id']: a for a in expected}


def configure_engine(config, cache, protocol):
    """Opt in only this collection's cached engines, including F3 warmup."""
    inventory(protocol)
    extra = config['kv_transfer_config']['kv_connector_extra_config']
    if protocol.get('hardware_profile') == 'l20-tp2' and protocol.get('tp') == 2:
        # Qwen3-32B TP2: each buffer holds one 8 MiB all-layer KV block.
        # 1152 covers the <=1024 context blocks while reserving 9 GiB/rank,
        # instead of the native default's 32 GiB/rank (320 GiB for five pairs).
        for connector in extra['ucm_connectors']:
            connector['ucm_connector_config']['buffer_number'] = 1152
    extra['tree_f3'] = dict(protocol=PROTOCOL, cache_dir=str(cache), namespace=str(cache), readonly=False)
    for sparse in extra['ucm_sparse_config'].values():
        sparse.update(method=METHOD, ratio=min(protocol['percentages'])/100)


def f3_snapshot(cache, samples):
    from runner.tree_prophetkv import snapshot_f3
    result = snapshot_f3(cache, str(cache), samples)
    folder = cache/'tree-f3'
    if folder.is_symlink() or {p.name for p in folder.iterdir()} != set(result):
        raise RuntimeError('Unexpected F3 files in owned prompt cache')
    return result


def retire_warmup(cache, tokens):
    """Called after all TP ranks retire the known construction warmup."""
    from runner.tree_prophetkv import f3_path, read_f3
    read_f3(cache, str(cache), tokens)
    path = f3_path(cache, tokens)
    if set(path.parent.iterdir()) != {path}:
        raise RuntimeError('Unexpected warmup F3 files')
    path.unlink()


from runner.sweep import PromptCache


class TreePromptCache(PromptCache):
    """Bound F3 lifetime to the same owned, retired prompt as its immutable KV."""
    def __init__(self, cache, model, tp, namespace, sample):
        super().__init__(cache, model, tp, namespace, sample)
        self.sample = sample
        self.f3_before = None

    def ready(self):
        result = super().ready()
        self.f3_before = f3_snapshot(self.cache, [self.sample])
        return dict(result, tree_f3=self.f3_before)

    def unchanged(self):
        super().unchanged()
        if self.f3_before is None or self.f3_before != f3_snapshot(self.cache, [self.sample]):
            raise RuntimeError('Original F3 changed during inference')

    def delete(self):
        # The shared worker calls end only after answer retirement/publication.
        result = super().delete()
        for name in self.f3_before:
            (self.cache/'tree-f3'/name).unlink()
        return dict(result, deleted_f3_records=len(self.f3_before))
