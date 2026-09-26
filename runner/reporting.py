"""Answer-only scoring and live/final collection reports (CPU only)."""
import csv
import io
import math
from pathlib import Path
import re
import statistics
import time

SCORERS = ('exact_match', 'choice', 'reference_coverage', 'contains_any', 'longbench_v2')
EVALUATION_FIELDS = {'subtask', 'references', 'scoring'}


def evaluation_metadata(source, overrides=None):
    """References are reporting metadata and never enter model input/selection."""
    nested = source.get('evaluation', {})
    values = dict(subtask=source.get('subtask', source.get('task', 'unlabeled')),
                  references=source.get('references', source.get('answers', [])),
                  scoring=source.get('scoring', 'exact_match'))
    values.update(nested)
    values.update({k: v for k, v in (overrides or {}).items() if k in EVALUATION_FIELDS})
    if set(values) != EVALUATION_FIELDS:
        raise ValueError('Unknown evaluation metadata field')
    if not isinstance(values['subtask'], str) or not values['subtask'].strip():
        raise ValueError('subtask must be a nonempty string')
    refs = values['references']
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list) or any(not isinstance(r, str) or not r.strip() for r in refs):
        raise ValueError('references must contain nonempty strings')
    values['references'] = refs
    if values['scoring'] not in SCORERS:
        raise ValueError(f'scoring must be one of {SCORERS}')
    if values['scoring'] in ('choice', 'longbench_v2') and any(r.strip().upper() not in 'ABCD' or len(r.strip()) != 1 for r in refs):
        raise ValueError('choice references must be A, B, C or D')
    return values


def normalized(text):
    return ' '.join(text.casefold().split())


def extract_choice(answer):
    # Prefer an explicit final-answer statement, then an unambiguous standalone
    # choice. Never search the thinking channel or guess among several letters.
    explicit = re.findall(r'\b(?:final\s+answer|answer)\s*(?:is|:|=)\s*\(?([A-D])\)?\b', answer, re.I)
    if explicit:
        return explicit[-1].upper()
    matches = re.findall(r'(?<!\w)\(?([A-D])\)?(?!\w)', answer)
    return matches[0] if len(set(matches)) == 1 else None


def score_answer(answer, evaluation):
    references, scorer = evaluation['references'], evaluation['scoring']
    if not references:
        return None
    prediction = normalized(answer)
    refs = [normalized(r) for r in references]
    if scorer == 'exact_match':
        return float(prediction in refs)
    if scorer == 'choice':
        return float(extract_choice(answer) in [r.strip().upper() for r in references])
    if scorer == 'longbench_v2':
        # Official LongBench v2 pred.py: remove asterisks, prefer parenthesized
        # exact-case answer statements, then unparenthesized statements.
        text = answer.replace('*', '')
        extracted = None
        for pattern in (r'The correct answer is \(([A-D])\)', r'The correct answer is ([A-D])'):
            match = re.search(pattern, text)
            if match:
                extracted = match.group(1)
                break
        return float(extracted in references)
    if scorer == 'contains_any':
        return float(any(r in prediction for r in refs))
    if scorer == 'reference_coverage':
        return sum(r in prediction for r in refs) / len(refs)
    raise ValueError(f'Unknown scorer: {scorer}')


class OutputAnalyzer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.open = tokenizer.convert_tokens_to_ids('<think>')
        self.close = tokenizer.convert_tokens_to_ids('</think>')
        for token, text in ((self.open, '<think>'), (self.close, '</think>')):
            if type(token) is not int or tokenizer.convert_ids_to_tokens(token) != text:
                raise ValueError('Expected Qwen3 thinking delimiter tokens')
        self.controls = set(tokenizer.all_special_ids) | {self.open, self.close}

    def analyze(self, token_ids, prompt_ids, thinking):
        # Native thinking usually starts in the chat template, before generation.
        in_thinking = thinking
        for token in reversed(prompt_ids):
            if token in (self.open, self.close):
                in_thinking = token == self.open
                break
        reason, answer, controls = [], [], 0
        for token in token_ids:
            if token == self.open:
                in_thinking = True
            elif token == self.close:
                in_thinking = False
            if token in self.controls:
                controls += 1
            else:
                (reason if in_thinking else answer).append(token)
        return dict(thinking_tokens=len(reason), answer_tokens=len(answer),
                    control_tokens=controls, output_tokens=len(token_ids),
                    unfinished_thinking=in_thinking,
                    answer_text=self.tokenizer.decode(answer, skip_special_tokens=False))


def metrics(records, expected):
    ttft = [r['timings']['ttft_seconds'] for r in records]
    scores = [r['accuracy'] for r in records if r['accuracy'] is not None]
    result = dict(expected=expected, completed=len(records), accuracy_scored=len(scores),
                  accuracy_unscored=len(records)-len(scores),
                  accuracy_percent=100*statistics.mean(scores) if scores else None,
                  mean_ttft_seconds=statistics.mean(ttft) if ttft else None,
                  median_ttft_seconds=statistics.median(ttft) if ttft else None,
                  output_cap_reached=sum(r['output_cap_reached'] for r in records),
                  unfinished_thinking=sum(r['unfinished_thinking'] for r in records))
    for name in ('thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens'):
        values = [r[name] for r in records]
        result['mean_'+name] = statistics.mean(values) if values else None
        result['total_'+name] = sum(values)
    return result


def aggregate(records, expected, context, status):
    """Overall means are prompt-weighted; missing reference scores are excluded."""
    expected_map = {entry['prompt_id']: entry['subtask'] for entry in expected}
    if len(expected_map) != len(expected):
        raise ValueError('Duplicate expected prompt IDs')
    seen = set()
    for record in records:
        pid = record['prompt_id']
        if pid in seen or pid not in expected_map or record['subtask'] != expected_map[pid]:
            raise ValueError('Duplicate/unexpected result or subtask mismatch')
        seen.add(pid)
        score = record['accuracy']
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Accuracy must be null or a score in [0,1]')
        value = record['timings']['ttft_seconds']
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid TTFT')
        if any(type(record[k]) is not int or record[k] < 0 for k in
               ('thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens')):
            raise ValueError('Invalid token counts')
        if record['thinking_tokens']+record['answer_tokens']+record['control_tokens'] != record['output_tokens']:
            raise ValueError('Output token accounting mismatch')
    if status == 'completed' and len(records) != len(expected):
        raise ValueError('Cannot finalize an incomplete collection')
    subtasks = {task: metrics([r for r in records if r['subtask'] == task],
                             sum(t == task for t in expected_map.values()))
                for task in sorted(set(expected_map.values()))}
    return dict(schema_version=1, status=status, updated_at=time.time(), **context,
                overall=metrics(records, len(expected)), subtasks=subtasks,
                definitions=dict(ttft='seconds from measured submission to first generated token',
                    accuracy='100 × mean answer-only score over prompts with references',
                    overall='prompt-weighted, not an unweighted mean of subtasks',
                    lengths='generated content tokens; delimiters and special tokens counted separately'))


COLUMNS = ('expected', 'completed', 'accuracy_scored', 'accuracy_unscored', 'accuracy_percent',
           'mean_ttft_seconds', 'median_ttft_seconds', 'mean_thinking_tokens', 'mean_answer_tokens',
           'mean_control_tokens', 'mean_output_tokens', 'total_thinking_tokens', 'total_answer_tokens',
           'total_control_tokens', 'total_output_tokens', 'output_cap_reached', 'unfinished_thinking')


def render_markdown(report):
    lines = [f"Status: {report['status']} | Method: {report['method']} | Completed: "
             f"{report['overall']['completed']}/{report['overall']['expected']}", '',
             '| Subtask | Done/total | Scored | Accuracy (%) | Mean TTFT (s) | Median TTFT (s) | Mean thinking tokens | Mean answer tokens |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'N/A' if value is None else f'{value:.3f}'
    for label, row in [('Overall', report['overall']), *report['subtasks'].items()]:
        label = label.replace('|', '\\|').replace('\n', ' ')
        cells = [label, f"{row['completed']}/{row['expected']}", str(row['accuracy_scored'])]
        cells += [fmt(row[k]) for k in ('accuracy_percent', 'mean_ttft_seconds', 'median_ttft_seconds',
                                        'mean_thinking_tokens', 'mean_answer_tokens')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.extend(['', 'Overall is prompt-weighted. Accuracy uses only scored prompts; N/A means no references/results.',
                  'Thinking/answer lengths count content tokens; control tokens are separate in JSON/CSV.',
                  'Only validated, retired measured requests are included; warmup and priming are excluded.', ''])
    return '\n'.join(lines)


class AggregationReporter:
    def __init__(self, output, expected, context):
        self.output, self.expected, self.context = Path(output), expected, context
        self.records = []
        self.publish('live', 'initializing')

    def publish(self, kind, status, error=None):
        from runner.setups import atomic_json
        report = aggregate(self.records, self.expected, self.context, status)
        if error is not None:
            report['error'] = str(error)
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['scope', 'subtask', *COLUMNS])
        for scope, label, row in [('overall', '', report['overall'])] + [
                ('subtask', task, row) for task, row in report['subtasks'].items()]:
            writer.writerow([scope, label, *[row[k] for k in COLUMNS]])
        for suffix, content in (('md', render_markdown(report)), ('csv', stream.getvalue())):
            path = self.output / f'{kind}_aggregation.{suffix}'
            temp = path.with_suffix(path.suffix+'.tmp')
            temp.write_text(content)
            temp.replace(path)
        # JSON is the authoritative atomic snapshot, published last.
        atomic_json(self.output / f'{kind}_aggregation.json', report)
        return report

    def accept(self, record):
        candidate = [*self.records, record]
        aggregate(candidate, self.expected, self.context, 'running')
        self.records = candidate
        self.publish('live', 'running')

    def finish(self):
        self.publish('live', 'completed')
        return self.publish('final', 'completed')
