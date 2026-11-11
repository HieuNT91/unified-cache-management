"""Tree-only collection identity; separate from router action inventories."""
from runner.tree.prophetkv import METHOD, PROTOCOL

SCHEMA = 'tree-prophetkv-thinking-collection-v1'
RATIOS = (5, 20, 40, 60)


def chunk_size(protocol):
    value = protocol.get('kv_chunk_size')
    if value is not None and (type(value) is not int or value not in (512, 1024, 2048, 4096)):
        raise ValueError('Tree kv_chunk_size must be 512, 1024, 2048 or 4096')
    return value


def runtime_sample(sample, protocol):
    """Derive chunk metadata without modifying the prepared input or its hash."""
    size = chunk_size(protocol)
    if size is None:
        return sample  # Preserve existing experiments' exact prepared layout.
    from runner.inputs.layout import boundaries_for, validate_layout, token_hash
    validate_layout(sample['token_ids'], sample['boundaries'], sample['question_positions'], sparse=True)
    if sample.get('token_sha256') != token_hash(sample['token_ids']):
        raise ValueError('Prepared token identity changed')
    result = dict(sample, boundaries=boundaries_for(len(sample['token_ids']),
                  min(sample['question_positions']), size))
    validate_layout(result['token_ids'], result['boundaries'], result['question_positions'], sparse=True)
    return result


def actions(percentages):
    if (not percentages or len(set(percentages)) != len(percentages)
            or any(type(p) is not int or not 0 < p <= 100 for p in percentages)):
        raise ValueError('Expected distinct integer percentages in [1, 100]')
    return [dict(id=f'{METHOD}-{p}', method=METHOD, ratio=p/100) for p in percentages]


def inventory(protocol):
    chunk_size(protocol)
    expected = collection_actions(protocol)
    if (protocol.get('schema') != SCHEMA or protocol.get('tree_protocol') != PROTOCOL
            or protocol.get('actions') != expected
            or protocol.get('scheduled_actions') != [a['id'] for a in expected]
            or protocol.get('phase_order') != (['cached', 'baseline'] if protocol.get('followup_nocache') else ['cached'])
            or protocol.get('answer_validation') != 'native-answer-diagnostics-v1'):
        raise ValueError('Invalid tree-prophetkv collection protocol')
    return {a['id']: a for a in expected}


def collection_actions(protocol):
    from runner.router.adatree import METHOD, answer_generation
    from ucm.sparse.prophetkv.adatree import protocol_for
    from scripts.collection.collection_v3 import seeds
    size = chunk_size(protocol)
    selected_seeds = seeds(protocol)
    if (protocol.get('run_profile', protocol.get('reproduction')) != 'adatree-v1.1'
            or size not in (2048, 4096)
            or protocol.get('adaptive_routing') != protocol_for(size)
            or protocol.get('percentages') != ([5, 20, 40] if size == 4096 else [5, 20, 60])
            or protocol.get('dataset') not in ('ruler', 'longbench-v2')
            or (protocol['dataset'] == 'ruler' and selected_seeds != [0])
            or protocol.get('answer_generation') != answer_generation()
            or protocol.get('followup_nocache') is not True
            or protocol.get('baseline_source') is not None):
        raise ValueError('AdaTreeV1.1 requires its fixed chunk router and matching new no-cache answers')
    return [dict(id=METHOD, method=METHOD, ratio=None),
            dict(id='nocache', method='baseline', ratio=None)]


def configure_engine(config, cache, protocol):
    """Opt in only this collection's cached engines, including F3 warmup."""
    inventory(protocol)
    extra = config['kv_transfer_config']['kv_connector_extra_config']
    if protocol.get('run_profile', protocol.get('reproduction')) == 'adatree-v1.1':
        extra['adatree_chunk_size'] = protocol['kv_chunk_size']
    extra['tree_f3'] = dict(protocol=PROTOCOL, cache_dir=str(cache), namespace=str(cache), readonly=False)
    for sparse in extra['ucm_sparse_config'].values():
        sparse.update(method=METHOD, ratio=min(protocol['percentages'])/100)


def f3_snapshot(cache, samples):
    from runner.tree.prophetkv import snapshot_f3
    result = snapshot_f3(cache, str(cache), samples)
    folder = cache/'tree-f3'
    if folder.is_symlink() or {p.name for p in folder.iterdir()} != set(result):
        raise RuntimeError('Unexpected F3 files in owned prompt cache')
    return result


def retire_warmup(cache, tokens):
    """Called after all TP ranks retire the known construction warmup."""
    from runner.tree.prophetkv import f3_path, read_f3
    read_f3(cache, str(cache), tokens)
    path = f3_path(cache, tokens)
    if set(path.parent.iterdir()) != {path}:
        raise RuntimeError('Unexpected warmup F3 files')
    path.unlink()


from runner.runtime.prompt_cache import PromptCache


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
