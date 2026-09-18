"""Optional CPU-only RULER corpus setup; never downloads models or packages."""
import os
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
