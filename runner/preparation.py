"""CPU prompt preparation and model identity checks."""
import hashlib
import json
from pathlib import Path

from runner.config import validate_sample


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_model(path):
    cfg = json.loads((path / 'config.json').read_text())
    if (cfg.get('model_type'), cfg.get('num_hidden_layers'), cfg.get('hidden_size')) != ('qwen3', 64, 5120):
        raise ValueError('Use a local Qwen3-32B checkpoint')
    if cfg.get('quantization_config'):
        raise ValueError('Use the unquantized BF16 checkpoint')


def tokenize_sample(args, tokenizer=None):
    from transformers import AutoTokenizer
    check_model(args.model)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    question = args.question.read_text()
    if not question:
        raise ValueError('Empty question')
    messages = [{'role': 'user', 'content': args.context.read_text() + '\n\n' + question}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=args.thinking)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    start = rendered.rfind(question)
    if start < 0:
        raise ValueError('Question not preserved by chat template')
    question_indices = [i for i, (a, b) in enumerate(offsets)
                        if b > start and a < start + len(question)]
    if not question_indices:
        raise ValueError('Question token mapping is empty')
    from runner.layout import boundaries_for, stamp_sample
    sample = dict(token_ids=ids,
        boundaries=boundaries_for(len(ids), question_indices[0], getattr(args, 'kv_chunk_size', 4096)),
        question_positions=question_indices,
        thinking=args.thinking, max_output_tokens=args.max_output_tokens,
        model_config_sha256=digest(args.model / 'config.json'),
        context_sha256=digest(args.context), question_sha256=digest(args.question))
    stamp_sample(sample, tokenizer.chat_template)
    validate_sample(sample, getattr(args, 'kv_chunk_size', 4096), getattr(args, 'context_length', 131072))
    return sample


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sample = tokenize_sample(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, sample)
    print(f'Prepared {len(sample["token_ids"])} tokens')
