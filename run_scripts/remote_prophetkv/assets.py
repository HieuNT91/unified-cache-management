"""Optional CPU-only RULER corpus setup; never downloads models or packages."""
import os
import json
from pathlib import Path
import runpy
import shutil
import tempfile
import urllib.request


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Asset preparation must expose no GPUs')
    original_urlopen = urllib.request.urlopen
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    failures = []

    def fetch(url, *args, **kwargs):
        kwargs.setdefault('timeout', 45)
        try:
            return direct.open(url, *args, **kwargs)
        except Exception:
            try:
                return original_urlopen(url, *args, **kwargs)
            except Exception:
                failures.append(str(url))
                raise

    # The upstream downloader catches URL errors; track them to reject an
    # incomplete corpus rather than silently changing the evaluation data.
    urllib.request.urlopen = fetch
    directory = Path(os.environ['RULER_ROOT']).resolve() / 'scripts/data/synthetic/json'
    target = directory / 'PaulGrahamEssays.json'
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix='ruler-essays-') as temporary:
            previous = Path.cwd()
            try:
                shutil.copy2(directory / 'PaulGrahamEssays_URLs.txt', temporary)
                os.chdir(temporary)
                runpy.run_path(str(directory / 'download_paulgraham_essay.py'), run_name='__main__')
                if failures:
                    raise RuntimeError(f'Corpus download incomplete; no corpus installed. Failed URLs: {failures}')
                shutil.copy2(Path(temporary) / 'PaulGrahamEssays.json', target)
            finally:
                os.chdir(previous)
    sources = {
        'squad.json': ['https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'],
        'hotpotqa.json': [
            'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
            'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json'],
    }
    for name, urls in sources.items():
        destination = directory / name
        if destination.exists():
            continue
        error = None
        for url in urls:
            try:
                with fetch(url) as response:
                    data = response.read()
                parsed = json.loads(data)
                if name == 'squad.json':
                    assert parsed['data']
                else:
                    assert isinstance(parsed, list) and parsed and 'question' in parsed[0]
                temporary = destination.with_suffix('.json.tmp')
                temporary.write_bytes(data)
                temporary.replace(destination)
                break
            except Exception as exc:
                error = exc
        else:
            raise RuntimeError(f'Unable to prepare {name}: {error}')
    import nltk
    try:
        nltk.sent_tokenize('Check sentence tokenizer.')
    except LookupError:
        for resource in ('punkt', 'punkt_tab'):
            if not nltk.download(resource, raise_on_error=True):
                raise RuntimeError(f'Could not prepare NLTK {resource}')
        nltk.sent_tokenize('Check sentence tokenizer.')
    print(f'RULER assets ready: {target}')


if __name__ == '__main__':
    main()
