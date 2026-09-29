"""Server-local configuration and launcher wiring, without GPU work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServerEnvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {'PATH': os.environ['PATH']}

    def load(self, body, **overrides):
        return subprocess.run(['bash','-c',
            'set -euo pipefail; source "$1"; ucm_load_server_env "$2"; '+body,
            'loader',str(ROOT/'scripts/server_env.sh'),str(self.root)],
            env=dict(self.env,**overrides),capture_output=True,text=True)

    def test_default_file_quotes_expansion_and_exported_overrides(self):
        (self.root/'.env').write_text('# local config\nPROJECT="/file path"\n'
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
        (self.root/'.env').write_text('not-an-assignment PRIVATE_VALUE\n')
        result=self.load('echo should-not-run')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Expected KEY=VALUE',result.stderr)
        self.assertNotIn('PRIVATE_VALUE',result.stderr)

    def test_both_launchers_use_config_for_status_and_stop(self):
        executable=self.root/'fake python'
        executable.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
            'print(json.dumps(dict(argv=sys.argv[1:],gpu_a=os.environ.get("GPU_A"))))\n')
        executable.chmod(0o755)
        config=self.root/'server.env'
        config.write_text(f'PYTHON_BIN="{executable}"\nEXPERIMENT_DIR="/existing run"\n'
                          'PREPARED_DIR="/prepared inputs"\nGPU_A="GPU-custom"\n')
        for launcher in ('l20_ruler.sh','a800_longbench.sh'):
            for command in ('status','stop','counts'):
                with self.subTest(launcher=launcher,command=command):
                    result=subprocess.run(['bash',str(ROOT/'scripts'/launcher),command],cwd='/tmp',
                        env=dict(self.env,UCM_ENV_FILE=str(config)),capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    data=json.loads(result.stdout)
                    self.assertEqual(data['gpu_a'],'GPU-custom')
                    self.assertEqual(data['argv'][data['argv'].index('--output')+1],'/existing run')
                    expected={'stop':'sweep_control.py','status':'sweep_report.py','counts':'sweep_counts.py'}[command]
                    self.assertTrue(data['argv'][0].endswith(expected))
                    if command!='stop':
                        self.assertEqual(data['argv'][data['argv'].index('--manifest')+1],'/prepared inputs/manifest.jsonl')

    def test_both_launchers_pass_fast_validation_before_any_gpu_work(self):
        executable=self.root/'fake-python'
        executable.write_text('#!/usr/bin/env python3\nimport json,sys\n'
                              'print(json.dumps(sys.argv[1:]))\nsys.exit(17)\n')
        executable.chmod(0o755)
        config=self.root/'server.env'
        config.write_text(f'PYTHON_BIN="{executable}"\nEXPERIMENT_DIR="{self.root}/out"\n'
                          'PREPARED_DIR="/prepared inputs"\nRESUME_VALIDATION=fast\n')
        for launcher in ('l20_ruler.sh','a800_longbench.sh'):
            with self.subTest(launcher=launcher):
                result=subprocess.run(['bash',str(ROOT/'scripts'/launcher),'resume'],cwd='/tmp',
                    env=dict(self.env,UCM_ENV_FILE=str(config)),capture_output=True,text=True)
                self.assertEqual(result.returncode,17,result.stderr)
                args=json.loads(result.stdout)
                self.assertEqual(args[:2],['-m','runner.resume'])
                self.assertEqual(args[args.index('--validation')+1],'fast')

    def test_a800_single_group_waits_for_each_shard_and_never_uses_gpu_b(self):
        executable=self.root/'fake-python'
        executable.write_text(f'#!{sys.executable}\n'+'''import json,os,sys,time
from pathlib import Path
args=sys.argv[1:]
if args[0]=='-c':
    os.execv(sys.executable,[sys.executable,*args])
def event(kind,**data):
    with open(os.environ['EVENT_LOG'],'a') as out:
        out.write(json.dumps(dict(kind=kind,devices=os.environ.get('CUDA_VISIBLE_DEVICES'),**data))+'\\n')
if args[:2]==['-m','runner.resume']:
    root=Path(args[args.index('--output')+1])
    devices=args[args.index('--single-group-gpus')+1].split(',')
    assert args[args.index('--exclude-percentages')+1]=='15'
    (root/'continuation.json').write_text(json.dumps(dict(execution_mode='sequential',gpu_uuids=[devices,devices])))
    event('prepare')
elif args[:2]==['-m','runner.sweep']:
    shard=int(args[args.index('--shard')+1]);event('start',shard=shard)
    time.sleep(.1)
    if os.environ.get('FAIL_SHARD0')=='1' and shard==0:sys.exit(7)
    event('finish',shard=shard)
elif args[0].endswith('longbench_v2.py'):
    event('disk',groups=args[args.index('--groups')+1])
elif args[0].endswith('sweep_report.py'):
    event('report',final='--final' in args)
else:raise AssertionError(args)
''')
        executable.chmod(0o755)
        config=self.root/'server.env';log=self.root/'events'
        config.write_text(f'PYTHON_BIN="{executable}"\nEXPERIMENT_DIR="{self.root}/out"\n'
            'RESUME_SINGLE_GROUP=1\nRESUME_EXCLUDE_PERCENTAGES="15"\nGPU_A="surviving-group"\nGPU_B="withdrawn-group"\n')
        for fail in (False,True):
            if log.exists():log.unlink()
            result=subprocess.run(['bash',str(ROOT/'scripts/a800_longbench.sh'),'resume'],cwd='/tmp',
                env=dict(self.env,UCM_ENV_FILE=str(config),EVENT_LOG=str(log),FAIL_SHARD0=str(int(fail))),capture_output=True,text=True)
            self.assertEqual(result.returncode,1 if fail else 0,result.stderr)
            events=[json.loads(line) for line in log.read_text().splitlines()]
            workers=[(r['kind'],r['shard']) for r in events if 'shard' in r]
            self.assertEqual(workers,[('start',0)] if fail else [('start',0),('finish',0),('start',1),('finish',1)])
            self.assertTrue(all(r['devices']=='surviving-group' if 'shard' in r else r['devices']=='' for r in events))
            self.assertEqual(next(r['groups'] for r in events if r['kind']=='disk'),'1')
            self.assertEqual(events[-1]['final'],not fail)


if __name__=='__main__':unittest.main()
