#!/usr/bin/env python3
"""Collect per-method live/final sweep reports into one set of tables."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runner.reporting import COLUMNS, aggregate
from runner.setups import atomic_json, manifest_entries
from runner.sweep import configurations


def collect(output, manifest, percentages, final=False, shards=2):
    configs=configurations(percentages)
    rows=manifest_entries(manifest)
    expected=[dict(prompt_id=r['id'],subtask=r['subtask']) for r in rows]
    kind='final' if final else 'live'
    if final:
        for shard in range(shards):
            receipt=json.loads((output/f'group-{shard}'/'complete.json').read_text())
            if not receipt['engine_shutdown'] or not receipt['cache_deleted']:
                raise RuntimeError('Final summary requires engine exit and cache deletion')
        if sum(json.loads((output/f'group-{i}'/'complete.json').read_text())['measurements']
               for i in range(shards)) != len(expected)*len(configs):
            raise RuntimeError('Final measurement count differs from requested scope')
    reports={}
    fingerprints=set()
    for config in configs:
        path=output/config['name']/f'{kind}_aggregation.json'
        if path.exists():
            report=json.loads(path.read_text())
            if report['overall']['expected'] != len(expected):
                raise RuntimeError('Report scope differs from manifest')
            if (report['method'],report['ratio']) != (config['method'],config['ratio']):
                raise RuntimeError('Method/ratio mismatch')
            if final and (report['status'] != 'completed' or report['overall']['completed'] != len(expected)):
                raise RuntimeError('Final method report is incomplete')
            fingerprints.add(report['sweep_fingerprint'])
        elif final:
            raise RuntimeError(f'Missing final report: {path}')
        else:
            report=aggregate([],expected,dict(method=config['method'],ratio=config['ratio']),'pending')
        reports[config['name']]=report
    if len(fingerprints)>1:
        raise RuntimeError('Reports came from different sweep configurations')
    completed=sum(r['overall']['completed'] for r in reports.values())
    summary=dict(kind=kind,completed=completed,expected=len(expected)*len(configs),methods=reports)
    stream=io.StringIO(newline='');writer=csv.writer(stream)
    writer.writerow(['configuration','scope','subtask',*COLUMNS])
    lines=[f'{kind.title()} summary: {completed}/{summary["expected"]} validated measurements', '']
    tasks=sorted({r['subtask'] for r in expected})
    for task in [None,*tasks]:
        lines.extend([f'### {task or "Overall"}', '',
            '| Configuration | Done/total | Accuracy (%) | Mean TTFT (s) | Mean thinking tokens | Mean answer tokens |',
            '|---|---:|---:|---:|---:|---:|'])
        for name,report in reports.items():
            row=report['overall'] if task is None else report['subtasks'][task]
            writer.writerow([name,'overall' if task is None else 'subtask',task or '',*[row[k] for k in COLUMNS]])
            fmt=lambda value:'N/A' if value is None else f'{value:.3f}'
            values=[fmt(row[k]) for k in ('accuracy_percent','mean_ttft_seconds','mean_thinking_tokens','mean_answer_tokens')]
            lines.append('| '+ ' | '.join([name,f'{row["completed"]}/{row["expected"]}',*values])+' |')
        lines.append('')
    lines.extend(['TTFT excludes construction, warmup and priming. Lengths count generated content tokens.',
                  'Control tokens, output caps, median TTFT and scored denominators are in CSV/JSON.', ''])
    output.mkdir(parents=True,exist_ok=True)
    for suffix,content in [('md','\n'.join(lines)),('csv',stream.getvalue())]:
        path=output/f'{kind}_summary.{suffix}'
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(content);tmp.replace(path)
    atomic_json(output/f'{kind}_summary.json',summary)
    print(f'{kind}: {completed}/{summary["expected"]}; {output}/{kind}_summary.md')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--percentages',type=int,nargs='+',default=[1,5,10,15,20,30,40,60,80])
    parser.add_argument('--shards',type=int,default=2)
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    collect(args.output,args.manifest,args.percentages,args.final,args.shards)

if __name__=='__main__':main()
