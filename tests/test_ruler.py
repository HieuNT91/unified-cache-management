"""RULER source content validation and official scoring."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runner.reporting.aggregate import score_answer
from scripts.data import ruler
from scripts import run


class RulerTests(unittest.TestCase):
    def source_fixture(self, root):
        for name in ('scripts/synthetic.yaml','scripts/data/synthetic/constants.py'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
        path=root/'scripts/data/synthetic/json/english_words.json'
        path.parent.mkdir(parents=True)
        path.write_text('version https://git-lfs.github.com/spec/v1\n'
                        f'oid sha256:{ruler.ENGLISH_WORDS_SHA256}\nsize 1\n')
        return path

    def test_source_validation_does_not_require_git_or_a_matching_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.source_fixture(root)
            with patch.object(ruler.subprocess,'check_output',side_effect=AssertionError('No Git revision gate')):
                ruler.source_check(root)
                (root/'.git').mkdir();(root/'.git/HEAD').write_text('unrelated local revision\n')
                ruler.source_check(root)

    def test_source_validation_still_rejects_missing_files_and_changed_wordlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wordlist=self.source_fixture(root)
            wordlist.write_text('{"changed":"asset"}')
            with self.assertRaisesRegex(ValueError,'digest'):ruler.source_check(root)
            wordlist.unlink()
            with self.assertRaisesRegex(ValueError,'Missing RULER'):ruler.source_check(root)

    def test_run_checks_other_asset_content_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wordlist=self.source_fixture(root);expected={}
            for name in run.RULER_ASSETS:
                path=wordlist.with_name(name)
                if name!=wordlist.name:path.write_text(name)
                expected[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(run,'RULER_ASSETS',expected), \
                    patch.object(ruler.subprocess,'check_output',side_effect=AssertionError('No Git revision gate')):
                run.validate_ruler_assets(root)
                wordlist.with_name('squad.json').write_text('changed content')
                with self.assertRaisesRegex(ValueError,'asset differs'):run.validate_ruler_assets(root)


    def test_ruler_official_scoring_semantics(self):
        evaluation=dict(references=['A B','OTHER'],scoring='ruler_all')
        self.assertEqual(score_answer('a b',evaluation),.5)
        self.assertEqual(score_answer('a  b',evaluation),0.)  # no whitespace normalization
        evaluation['scoring']='ruler_any'
        self.assertEqual(score_answer('other',evaluation),1.)
        self.assertEqual(score_answer('',evaluation),0.)


if __name__=='__main__':unittest.main()
