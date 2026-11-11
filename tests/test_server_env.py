"""Server-local configuration and launcher wiring, without GPU work."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServerEnvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'config').mkdir()
        self.env = {'PATH': os.environ['PATH']}

    def load(self, body, **overrides):
        return subprocess.run(['bash','-c',
            'set -euo pipefail; source "$1"; ucm_load_server_env "$2"; '+body,
            'loader',str(ROOT/'scripts/launcher/server_env.sh'),str(self.root)],
            env=dict(self.env,**overrides),capture_output=True,text=True)

    def test_default_file_quotes_expansion_and_exported_overrides(self):
        (self.root/'config/.env').write_text('# local config\nPROJECT="/file path"\n'
            'PREPARED_DIR="$PROJECT/prepared"\nexport PYTHON_BIN="/file/python"\n')
        result=self.load('printf "%s\\n" "$PROJECT" "$PREPARED_DIR" "$PYTHON_BIN"',
                         PROJECT='/override path',PYTHON_BIN='/override/python')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.splitlines(),['/override path','/override path/prepared','/override/python'])

    def test_missing_default_keeps_legacy_defaults_but_explicit_missing_fails(self):
        result=self.load('echo ready')
        self.assertEqual(result.stdout,'ready\n')
        result=self.load('echo should-not-run',UCM_ENV_FILE='missing.env')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Missing UCM_ENV_FILE',result.stderr)
        self.assertEqual(result.stdout,'')

    def test_custom_file_crlf_and_no_final_newline(self):
        (self.root/'custom.env').write_bytes(b'\r\n# comment\r\nEXPERIMENT_DIR="/path with spaces"')
        result=self.load('printf "%s" "$EXPERIMENT_DIR"',UCM_ENV_FILE='custom.env')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'/path with spaces')

    def test_invalid_line_fails_without_disclosing_value(self):
        (self.root/'config/.env').write_text('not-an-assignment PRIVATE_VALUE\n')
        result=self.load('echo should-not-run')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Expected KEY=VALUE',result.stderr)
        self.assertNotIn('PRIVATE_VALUE',result.stderr)








if __name__=='__main__':unittest.main()
