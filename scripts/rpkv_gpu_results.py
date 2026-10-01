#!/usr/bin/env python3
"""Independently verify and summarize the requested 210 GPU answers."""
import argparse
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runner.setups import atomic_json,file_hash
from runner.reporting import OutputAnalyzer,score_answer,scoring_text,metrics
from runner.corpus_records import accepted
from runner.layout import token_hash
from scripts.ruler_64000 import TASKS,definitions


def report(base,ruler):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Reporting is CPU-only')
    from transformers import AutoTokenizer
    source=ast.parse((ruler/'scripts/eval/evaluate.py').read_text())
    fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='postprocess_pred')
    namespace={'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original evaluator>','exec'),namespace)
    spec=importlib.util.spec_from_file_location('rpkv_original_metrics',ruler/'scripts/eval/synthetic/constants.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    custom,_=definitions(ruler);summary={};details=[];pins={};audits={}
    for dataset,count in [('ruler',65),('longbench-v2',5)]:
        root=base/'results'/dataset;protocol=json.loads((root/'protocol.json').read_text());rows=json.loads((root/'rows.json').read_text())
        assert len(rows)==count and (root/'complete.json').exists()
        if dataset=='ruler':assert {t:sum(r['subtask']==t for r in rows) for t in TASKS}==dict.fromkeys(TASKS,5)
        for path in (root/'processes').glob('*-exit.json'):assert json.loads(path.read_text())['owned_engines_exited']
        tokenizer=AutoTokenizer.from_pretrained(protocol['model'],local_files_only=True);analyzer=OutputAnalyzer(tokenizer)
        bymethod={a['id']:[] for a in protocol['actions']}
        for row in rows:
            sample=json.loads((Path(protocol['prepared'])/row['prepared']).read_text())
            assert token_hash(sample['token_ids'])==sample['token_sha256']
            if dataset=='longbench-v2':assert not sample['truncation']['truncated'] and sample['max_output_tokens']==16384
            probe=accepted(root,'probe',row,protocol,full=True);assert probe is not None
            for action in protocol['actions']:
                result=accepted(root,action['id'],row,protocol,full=True);assert result is not None
                assert result['token_sha256']==sample['token_sha256'] and result['max_output_tokens']==sample['max_output_tokens']
                analysis=analyzer.analyze(result['output_token_ids'],sample['token_ids'],sample['thinking'])
                assert all(result[k]==v for k,v in analysis.items())
                assert score_answer(scoring_text(result['prediction'],analysis,result),result)==result['accuracy']
                if dataset=='ruler':assert result['thinking_tokens']==0
                bymethod[action['id']].append(result)
                details.append(dict(dataset=dataset,prompt_id=row['id'],task=row['subtask'],method=action['id'],input_tokens=len(sample['token_ids']),output_cap=sample['max_output_tokens'],accuracy=result['accuracy'],ttft_seconds=result['timings']['ttft_seconds'],engine_ttft_seconds=result['timings']['answer_engine_ttft_seconds'],thinking_tokens=result['thinking_tokens'],answer_tokens=result['answer_tokens'],control_tokens=result['control_tokens'],output_cap_reached=result['output_cap_reached'],token_sha256=sample['token_sha256'],prediction=result['prediction']))
        summary[dataset]={}
        for action,records in bymethod.items():
            summary[dataset][action]=dict(overall=metrics(records,count),tasks={t:metrics([r for r in records if r['subtask']==t],sum(r['subtask']==t for r in rows)) for t in sorted({r['subtask'] for r in rows})})
            if dataset=='ruler':
                for t,m in summary[dataset][action]['tasks'].items():
                    group=[r for r in records if r['subtask']==t]
                    official=original.TASKS[custom[t]['task']]['metric_fn']([namespace['postprocess_pred'](r['prediction'],{}) for r in group],[r['references'] for r in group])
                    assert official==m['ruler_score']
        for path in root.glob('records/*/*/validated.json'):pins[str(path.relative_to(base))]=file_hash(path)
        audits[dataset]=dict(answers=count*3,independent_probes=count,same_original_token_hash=True,all_rank_replay=True,retirement_and_cache_immutability=True,independent_scoring_equal=True)
    output=base/'final';output.mkdir(exist_ok=False)
    atomic_json(output/'report.json',summary);atomic_json(output/'validation.json',dict(passed=True,answers=210,probes=70,audits=audits,accepted_receipts=pins))
    with (output/'measurements.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    lines=['# rpkv GPU validation','', 'Qwen3-32B BF16, TP4 on physical RTX 4500 Ada GPUs1–4; native YaRN4. All methods use identical original prepared tokens and metadata-only chunk boundaries.','',
        'RULER: 13 tasks ×5, seed42 task batches, original 65536 total window and task caps; non-thinking. LongBench v2: five seed42 random inputs from the explicitly approved full-text fitting pool, native thinking and16384 output cap; no truncation. See prepared/longbench-v2-fitting/selection.json for the full eligibility inventory.','',
        'TTFT is measured submission-to-first-token plus TP action synchronization. Offline cache construction, readiness, ordinary priming and independent diagnostic probe time are separate. Thinking/answer lengths count newly generated content tokens; special/control tokens are separate. Sparse controls ran before connector-free baselines.','',
        '| Dataset / task | Method | N | Accuracy % | Mean TTFT s | Mean thinking tokens | Mean answer tokens | Capped |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset,methods in summary.items():
        for task in ['overall']+sorted(next(iter(methods.values()))['tasks']):
            for action,values in methods.items():
                m=values['overall'] if task=='overall' else values['tasks'][task]
                lines.append(f"| {dataset} / {task} | {action} | {m['completed']} | {m['accuracy_percent']:.2f} | {m['mean_ttft_seconds']:.4f} | {m['mean_thinking_tokens']:.2f} | {m['mean_answer_tokens']:.2f} | {m['output_cap_reached']} |")
    lines+=['','All210 answers and70 independent probes passed the recorded GPU acceptance gates. This validates the measured inputs and methods; the separate full-repair/dense-fallback and synthetic edge-case GPU matrix is not implied by these measurements. Five samples per task provide descriptive comparisons only.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--ruler',type=Path,required=True)
    args=parser.parse_args();report(args.root.resolve(),args.ruler.resolve())
if __name__=='__main__':main()
