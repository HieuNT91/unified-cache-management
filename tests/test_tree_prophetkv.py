"""CPU references for F3, hierarchical need and original-position masks."""
import copy
import math
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
import torch._dynamo  # Initialize before fixtures snapshot/restore sys.modules.

import runner.generation  # Load before fixtures temporarily replace sys.modules.
from ucm.sparse.prophetkv.tree_selection import F3Recorder, tree_need
from ucm.sparse.prophetkv import tree_attention


class TreeMathTests(unittest.TestCase):
    def test_density_matches_recursive_tree_including_equal_links(self):
        rng = np.random.default_rng(42)
        for n in (1, 2, 7, 64, 127):
            for tied in (False, True):
                s = rng.integers(0, 4, n).astype(float) if tied else rng.random(n)
                s[0] = -np.inf
                u = rng.random(n); u /= u.sum()
                reference = u.copy()
                def visit(a, z):
                    reference[a:z] = np.maximum(reference[a:z], u[a:z].mean())
                    if z-a > 1:
                        k = a+1+int(np.argmin(s[a+1:z])); visit(a, k); visit(k, z)
                visit(0, n)
                np.testing.assert_allclose(tree_need(s, u), reference, atol=1e-12)

    def test_chunk_boundary_ties_preserve_upstream_outputs(self):
        # Deliberately retained at the user's request: this differs from full
        # recursive ancestor inheritance for the final two tokens (1/6, not 1/4).
        s = [-np.inf, 1., -np.inf, 1., -np.inf, 1.]
        u = [0., 0., 0., 1., 0., 0.]
        np.testing.assert_allclose(tree_need(s, u), [1/6, 1/6, .5, 1., 1/6, 1/6])

    def test_bad_density_inputs_fail(self):
        for s, u in [([], []), ([1], [1, 2]), ([np.nan], [1]), ([np.inf], [1]), ([0], [-1]), ([0], [np.nan])]:
            with self.subTest(s=s, u=u), self.assertRaises(ValueError):
                tree_need(s, u)

    def test_f3_matches_direct_local_window_mass_and_tp_head_weighting(self):
        torch.manual_seed(42)
        q, k = torch.randn(64, 4, 8), torch.randn(64, 2, 8)
        reference = torch.zeros(64)
        for h in range(4):
            logits = q[:, h] @ k[:, h//2].T / math.sqrt(8)
            logits.masked_fill_(torch.ones(64, 64, dtype=torch.bool).triu(1), -float('inf'))
            logits[:, :4] = -float('inf'); logits[:4] = 0
            weights = logits.softmax(-1)
            for t in range(64):
                numerator = denominator = 0.
                for i in range(t, min(t+16, 64)):
                    numerator += weights[i, max(4, t-16):max(4, t)].sum().item()
                    denominator += weights[i, max(4, t-16):max(4, i)].sum().item()
                reference[t] += numerator / max(denominator, 1e-9) / 4
        recorders = []
        for rank in range(2):
            rec = F3Recorder(tuple(range(64)), 8, 4)
            self.assertTrue(rec.add_layer(0, q[:, rank*2:rank*2+2], k[:, rank:rank+1]))
            recorders.append(rec)
        total = recorders[0].acc + recorders[1].acc
        for rec in recorders:
            def reduce(t):
                return torch.tensor([4.]) if t.numel() == 1 else total
            with patch.dict('sys.modules', {'vllm.distributed': NS(get_tp_group=lambda: NS(world_size=2), tensor_model_parallel_all_reduce=reduce)}):
                np.testing.assert_allclose(rec.result(), reference.numpy(), atol=2e-6)
        self.assertEqual(F3Recorder(tuple(range(64)), 8, 64).L, 16)
        self.assertEqual(F3Recorder(tuple(range(64)), 8, 36).L, 9)

    def test_f3_rejects_missing_and_duplicate_layers(self):
        rec = F3Recorder(tuple(range(64)), 8, 8)
        q = torch.zeros(64, 1, 8)
        rec.add_layer(0, q, q)
        with self.assertRaises(RuntimeError): rec.add_layer(0, q, q)
        with patch.dict('sys.modules', {'vllm.distributed': NS()}), self.assertRaises(RuntimeError): rec.result()


class MaskTests(unittest.TestCase):
    @staticmethod
    def attention(q, k, v, positions, scale):
        logits = torch.einsum('qhd,khd->hqk', q.float(), k.float().repeat_interleave(q.shape[1]//k.shape[1], 1)) * scale
        if positions is not None:
            logits.masked_fill_(torch.arange(len(k))[None, None, :] > positions[None, :, None], -float('inf'))
        out = torch.einsum('hqk,khd->qhd', logits.softmax(-1), v.float().repeat_interleave(q.shape[1]//v.shape[1], 1))
        return out, logits.logsumexp(-1)

    def test_gathered_recompute_and_two_pass_suffix_match_dense_mask(self):
        torch.manual_seed(7)
        b = [0, 64, 128, 192, 448]
        selected = torch.tensor([64, 65, 90, 127, 128, 140, 190])
        k, v = torch.randn(448, 2, 8), torch.randn(448, 2, 8)
        calls = []
        def kernel(q, kk, vv, out, qs, qn, ks, kn, pos, maxq, **kwargs):
            calls.append(len(qn))
            for a, n, start, length in zip(qs, qn, ks, kn):
                out[a:a+n] = self.attention(q[a:a+n], kk[start:start+length], vv[start:start+length], pos[a:a+n], kwargs['sm_scale'])[0]
        def flash(q, kk, vv, *, causal, softmax_scale, **kwargs):
            pos = torch.arange(len(kk)-len(q), len(kk)) if causal else None
            return self.attention(q, kk, vv, pos, softmax_scale)
        for pos in (selected[:4], torch.cat((selected, torch.arange(192, 240))), torch.arange(240, 300)):
            q = torch.randn(len(pos), 4, 8); out = torch.full_like(q, float('nan'))
            allowed = torch.cat((torch.arange(64), selected))
            with patch('ucm.sparse.prophetkv.attention.positional_kernel', return_value=kernel), patch.dict('sys.modules', {'vllm.vllm_flash_attn': NS(flash_attn_varlen_func=flash)}):
                tree_attention.rmask_attention(q, k, v, pos, b, selected, 8**-.5, out)
                tree_attention.masked_suffix_attention(q, k, v, pos, b, allowed, 8**-.5, out)
            ref = tree_attention.dense_reference(q, k, v, pos, b, selected)
            torch.testing.assert_close(out, ref, atol=2e-6, rtol=2e-6)
        self.assertEqual(calls, [1, 2])

    def test_many_chunks_bound_kv_workspace_and_preserve_all_query_rows(self):
        # The L20 failure occurs in a late 16K prefill step with 1024-token
        # chunks. Keep its 64K layout, but use scalar K/V to run on CPU.
        bounds = list(range(0, 64513, 1024)) + [64768]
        candidates = torch.arange(1024, 64512)
        selected = candidates[(candidates % 5) < 3]  # approximately 60%
        positions = selected[selected >= 50176]
        key = torch.arange(64512, dtype=torch.float32)[:, None, None]
        value = -key
        query = positions.float()[:, None, None]
        output = torch.full_like(query, float('nan'))
        calls = []; covered = torch.zeros(len(positions), dtype=torch.int32)

        def kernel(q, k, v, out, qs, qn, ks, kn, qpos, maxq, **kwargs):
            self.assertLessEqual(len(k), len(key), 'RMASK duplicated too much K/V at once')
            self.assertEqual(len(v), len(k))
            # Passing the full Q tensor preserves the existing tile heuristic.
            self.assertEqual(len(q), len(query))
            calls.append(len(k))
            for a, n, start, length in zip(qs.tolist(), qn.tolist(), ks.tolist(), kn.tolist()):
                p = positions[a:a+n]
                chunk = int(p[0]) // 1024
                expected = torch.cat((torch.arange(1024),
                    selected[selected < chunk*1024], torch.arange(chunk*1024, int(p[-1])+1)))
                torch.testing.assert_close(k[start:start+length, 0, 0], expected.float())
                torch.testing.assert_close(v[start:start+length], -k[start:start+length])
                torch.testing.assert_close(expected[qpos[a:a+n]], p)
                covered[a:a+n] += 1
                out[a:a+n] = q[a:a+n]

        with patch('ucm.sparse.prophetkv.attention.positional_kernel', return_value=kernel):
            tree_attention.rmask_attention(query, key, value, positions, bounds,
                                           selected, 1., output)
        self.assertGreater(len(calls), 1)
        self.assertTrue(torch.equal(covered, torch.ones_like(covered)))
        torch.testing.assert_close(output, query)

    def test_batched_recompute_matches_dense_mask_and_preserves_suffix(self):
        torch.manual_seed(29)
        bounds = list(range(0, 641, 64)) + [704]
        candidates = torch.arange(64, 640)
        for ratio in (.05, .2, .4, .6, 1.):
            with self.subTest(ratio=ratio):
                selected = candidates[torch.randperm(len(candidates))[:int(ratio*len(candidates))]].sort().values
                # Only a late prefill step is active; prior fresh keys are still visible.
                positions = torch.cat((selected[selected >= 256], torch.arange(640, 657)))
                key, value = torch.randn(657, 2, 8), torch.randn(657, 2, 8)
                query = torch.randn(len(positions), 4, 8)
                output = torch.full_like(query, 123.)
                key_before, value_before = key.clone(), value.clone()
                calls = []

                def kernel(q, k, v, out, qs, qn, ks, kn, qpos, maxq, **kwargs):
                    self.assertLessEqual(len(k), len(key))
                    calls.append(len(k))
                    for a, n, start, length in zip(qs, qn, ks, kn):
                        out[a:a+n] = self.attention(q[a:a+n], k[start:start+length],
                            v[start:start+length], qpos[a:a+n], kwargs['sm_scale'])[0]

                with patch('ucm.sparse.prophetkv.attention.positional_kernel', return_value=kernel):
                    tree_attention.rmask_attention(query, key, value, positions, bounds,
                                                   selected, 8**-.5, output)
                context = positions < 640
                reference = tree_attention.dense_reference(query[context], key, value,
                                                           positions[context], bounds, selected)
                torch.testing.assert_close(output[context], reference, atol=2e-6, rtol=2e-6)
                self.assertTrue((output[~context] == 123.).all())
                torch.testing.assert_close(key, key_before)
                torch.testing.assert_close(value, value_before)
                self.assertGreater(len(calls), 1)

    def test_batched_offsets_with_real_kernel_cpu_interpreter(self):
        import importlib.util
        import os
        from pathlib import Path
        import subprocess
        import sys
        if not importlib.util.find_spec('triton'):
            self.skipTest('CPU interpreter requires Triton')
        import triton
        if triton.__version__ == '3.3.0' and np.lib.NumpyVersion(np.__version__) >= '2.4.0':
            self.skipTest('Triton 3.3 interpreter scalar conversion requires NumPy < 2.4')
        script = '''
import torch
from types import SimpleNamespace
from ucm.sparse.prophetkv.tree_attention import rmask_attention, dense_reference
torch.manual_seed(29)
bounds = [0,64,128,192,256,320,384,448]
for step in (2, 1):
    selected = torch.arange(64,384,step)
    positions = selected[selected >= (128 if step == 2 else 64)]
    query = torch.randn(len(positions),4,16)
    key = torch.randn(384,2,16); value = torch.randn_like(key)
    reference = dense_reference(query,key,value,positions,bounds,selected)
    for name in ('fa2-ragged-v1', '_ragged_positions_fa2_kernel', 'original'):
        output = torch.full_like(query,float('nan'))
        rmask_attention(query,key,value,positions,bounds,selected,16**-.5,output,
                        sparse=SimpleNamespace(position_attention_kernel=name))
        torch.testing.assert_close(output,reference,atol=2e-5,rtol=2e-4)
print('six batched masked-attention/GQA interpreter cases passed; CPU only')
'''
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-c', script], cwd=root,
            capture_output=True, text=True, timeout=60,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', TRITON_INTERPRET='1', PYTHONPATH=str(root)))
        self.assertEqual(result.returncode, 0, result.stderr)


class SelectionTests(unittest.TestCase):
    def test_s4_is_inside_exact_budget_and_ties_are_stable(self):
        from ucm.sparse.prophetkv.tree_selection import select_tree
        from runner.tree_prophetkv import budget
        bounds = [0, 64, 128, 192, 448]
        for ratio in (8/128, .15, .2, 1.):
            count, forced = budget(bounds, ratio)
            chosen, need = select_tree(np.ones(128), np.ones(128), bounds, ratio)
            expected = set(forced)
            for p in range(64, 192):
                if len(expected) == count: break
                expected.add(p)
            self.assertEqual(chosen, sorted(expected))
            self.assertEqual(len(chosen), math.floor(128*ratio))
        for ratio in (0., .01, float('nan'), 1.1):
            with self.assertRaises(ValueError): budget(bounds, ratio)

    def test_config_and_policy_switch_isolation(self):
        from runner.config import engine_config, METHODS
        from runner.temporary import configure_sparse
        from ucm.sparse.prophetkv.layers import resolve_layers
        self.assertIn('tree-prophetkv', METHODS)
        with patch.dict('os.environ', {'UCM_TREE': '1', 'UCM_RMASK': '1'}):
            plain = engine_config('/model', 'prophetkv', cache_dir='/cache')
        self.assertNotIn('tree_f3', plain['kv_transfer_config']['kv_connector_extra_config'])
        tree = engine_config('/model', 'tree-prophetkv', cache_dir='/cache')
        self.assertFalse(tree['kv_transfer_config']['kv_connector_extra_config']['tree_f3']['readonly'])
        self.assertEqual(resolve_layers('tree-prophetkv'), tuple(range(64)))
        with self.assertRaises(ValueError): resolve_layers('tree-prophetkv', [63])
        with self.assertRaisesRegex(ValueError, 'fresh setup'):
            engine_config('/model', 'tree-prophetkv', cache_dir='/cache', persistent={'readonly': True})
        sparse = NS(request_state=NS(pending=None), active=False, connector=NS(worker_request_id=None, store=NS(pending={})))
        with self.assertRaisesRegex(ValueError, 'tree-enabled'): configure_sparse(sparse, 'tree-prophetkv', .2)
        sparse.tree_f3_config = {'enabled': True}
        configure_sparse(sparse, 'tree-prophetkv', .2)
        configure_sparse(sparse, 'prophetkv', .2)
        self.assertEqual(sparse.method, 'prophetkv')


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        from runner.tree_prophetkv import METHOD, PROTOCOL, budget
        from ucm.sparse.prophetkv.tree_selection import select_tree
        sample = dict(boundaries=[0, 64, 128, 384], question_positions=[200, 201])
        ratio = .2; b = sample['boundaries']
        count, forced = budget(b, ratio)
        chosen, _ = select_tree(np.ones(64), np.ones(64), b, ratio)
        event = dict(kind='prophetkv_selection', selection_method=METHOD, tree_protocol=PROTOCOL,
            scores=[1.]*64, selected_positions=chosen, selected_count=count, eligible_count=64, ratio=ratio,
            unit_budget=count, forced=forced, f3_links=[1.]*64, f3_layers=list(range(16)),
            question_positions=[200, 201], probe_question_positions=list(range(200, 384)),
            recompute_mask='prefix-fresh-earlier-own-chunk-causal', suffix_mask='prefix-fresh-causal-suffix',
            decode_mask='native-unmasked', scoring_layers=list(range(64)), fusion='mean_layers_fp32', alignment_count=64)
        positions = chosen+list(range(128,384))
        events = [event, dict(kind='prefill_step', start=64, end=384, scheduled_tokens=320,
            recomputed_tokens=len(positions), no_forward=False, prefill_complete=True)]
        events.extend(dict(kind='layer_counts', layer=f'model.layers.{i}.self_attn.attn', start=64, end=384,
            selected_positions=positions, selected_set_verified=True, projection_tokens=len(positions),
            attention_tokens=len(positions), ffn_tokens=len(positions)) for i in range(64))
        return sample, [dict(rank=r, diagnostics=copy.deepcopy(events)) for r in range(2)]

    def test_full_native_evidence_and_corruption(self):
        from runner.generation import verify_diagnostics
        sample, workers = self.fixture()
        verify_diagnostics(workers, sample, 'tree-prophetkv', .2, 2, range(64))
        for field, value in [('unit_budget', 16), ('forced', []), ('f3_links', [0.]), ('f3_layers', [0]),
                             ('probe_question_positions', [200, 201]), ('decode_mask', 'masked'),
                             ('tree_protocol', 'old'), ('selected_positions', list(range(65,77)))]:
            sample, workers = self.fixture(); workers[0]['diagnostics'][0][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_diagnostics(workers, sample, 'tree-prophetkv', .2, 2, range(64))
        sample, workers = self.fixture(); workers[1]['diagnostics'][0]['f3_links'] = [.5]*64
        with self.assertRaisesRegex(RuntimeError, 'TP ranks disagree on F3'):
            verify_diagnostics(workers, sample, 'tree-prophetkv', .2, 2, range(64))
        sample, workers = self.fixture()
        with self.assertRaises(RuntimeError): verify_diagnostics(workers, sample, 'prophetkv', .2, 2, range(64))
        with self.assertRaises(ValueError): verify_diagnostics(workers, sample, 'tree-prophetkv', .2, 2, range(64), gpu_scores=True)


class PersistentTreeTests(unittest.TestCase):
    def setUp(self):
        import test_setups
        self.fixture = test_setups.SetupTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.args.tree_prophetkv = True
        original = f.fake_build
        def fake_build(root, descriptor, samples, missing, args, progress):
            from runner.setups import chunks_of, fingerprint
            from runner.tree_prophetkv import write_f3
            original(root, descriptor, samples, missing, args, progress)
            for _, sample in samples:
                for tokens in chunks_of(sample):
                    if fingerprint(tokens) in missing:
                        write_f3(root/'cache', descriptor['fingerprint'], tokens, [.25]*len(tokens))
        f.fake_build = fake_build

    def test_fresh_setup_receipt_reuse_and_readonly_dry_run(self):
        import io, json
        from runner import setups
        from runner.tree_prophetkv import PROTOCOL, snapshot_f3
        f = self.fixture; descriptor, _ = f.build()
        self.assertEqual(descriptor['preparation']['tree_protocol'], PROTOCOL)
        receipt = setups.check_receipt(f.args.output, descriptor)
        samples = [s for _, s in setups.load_frozen(f.args.output, descriptor)]
        before = snapshot_f3(f.args.output/'cache', descriptor['fingerprint'], samples)
        self.assertEqual(receipt['tree_f3'], before)
        self.assertEqual(len(before), 2)
        f.build(); self.assertEqual(len(f.build_calls), 1)
        with patch('sys.stdout', new=io.StringIO()) as output:
            setups.run_collection(f.run_args(method='tree-prophetkv', ratio=.2, dry_run=True))
        cfg = json.loads(output.getvalue())['engine']['kv_transfer_config']['kv_connector_extra_config']
        self.assertTrue(cfg['tree_f3']['readonly'])
        self.assertEqual(cfg['tree_f3']['namespace'], descriptor['fingerprint'])
        self.assertEqual(snapshot_f3(f.args.output/'cache', descriptor['fingerprint'], samples), before)

    def test_missing_f3_resumes_without_rewriting_committed_kv(self):
        from runner import setups
        from runner.tree_prophetkv import f3_path
        f = self.fixture; descriptor, inventory = f.build()
        kv = setups.snapshot(f.args.output/'cache', inventory)
        tokens = setups.chunks_of(f.sample)[0]
        f3_path(f.args.output/'cache', tokens).unlink()
        f.build()
        self.assertEqual(f.build_calls[-1], [setups.fingerprint(tokens)])
        self.assertEqual(setups.snapshot(f.args.output/'cache', inventory), kv)

    def test_plain_setup_rejected_and_cannot_be_upgraded_in_place(self):
        from runner import setups
        f = self.fixture; f.args.tree_prophetkv = False; f.build()
        with self.assertRaisesRegex(ValueError, 'fresh setup'):
            setups.run_collection(f.run_args(method='tree-prophetkv', ratio=.2, dry_run=True))
        f.args.tree_prophetkv = True
        with self.assertRaisesRegex(ValueError, 'Incompatible preparation'): f.build()

    def test_resume_rejects_changed_f3_even_when_kv_is_incomplete(self):
        import json
        from runner import setups
        from runner.tree_prophetkv import f3_path
        f = self.fixture; descriptor, inventory = f.build()
        complete = (f.args.output/'complete.json').read_bytes()
        path = f3_path(f.args.output/'cache', setups.chunks_of(f.sample)[0])
        record = json.loads(path.read_text())
        record['payload']['values'] = [.75]*64
        record['sha256'] = setups.fingerprint(record['payload'])
        path.write_text(json.dumps(record))
        shard = f.args.output/'cache'/setups.required_files(inventory)[0]
        shard.write_bytes(b'incomplete')
        with self.assertRaisesRegex(RuntimeError, 'Committed F3 changed'): f.build()
        self.assertEqual((f.args.output/'complete.json').read_bytes(), complete)
        self.assertEqual(shard.read_bytes(), b'incomplete')

    def test_record_corruption_and_namespace_mismatch_fail_without_rewrite(self):
        from runner import setups
        from runner.tree_prophetkv import f3_path, read_f3, write_f3
        f = self.fixture; descriptor, _ = f.build()
        tokens = setups.chunks_of(f.sample)[0]; cache = f.args.output/'cache'
        with self.assertRaises(RuntimeError): read_f3(cache, 'another-setup', tokens)
        path = f3_path(cache, tokens); before = path.stat().st_mtime_ns
        write_f3(cache, descriptor['fingerprint'], tokens, [.75]*len(tokens))
        self.assertEqual(path.stat().st_mtime_ns, before)
        path.write_text('{}')
        with self.assertRaises(RuntimeError): f.build()
        with self.assertRaises(RuntimeError): read_f3(cache, descriptor['fingerprint'], tokens)
        self.assertEqual(path.read_text(), '{}')

class RuntimeTests(unittest.TestCase):
    def test_probe_suffix_s4_and_tp_preserve_original_metadata_and_stored_kv(self):
        import tempfile
        from ucm.sparse.prophetkv import runtime, attention
        from ucm.sparse.prophetkv.selection import RequestMetadata
        from runner.tree_prophetkv import write_f3
        torch.manual_seed(7)
        layers = [NS(self_attn=NS(o_proj=lambda x:(x,None), head_dim=2),
            post_attention_layernorm=lambda x,r:(x,r), mlp=lambda x:x) for _ in range(64)]
        aligned = set(); cache = torch.zeros(2,6,64,1,2); before = cache.clone()
        context = NS(virtual_engine=0, no_compile_layers={f'model.layers.{i}.self_attn.attn':NS(kv_cache=[cache]) for i in range(64)})
        seen = []; sink_seen = []; reductions = []; broadcasts = []
        def project(layer,h,r,positions):
            q = positions.float()[:,None,None].expand(-1,1,2)
            return q,q,q,h
        def importance(q,k):
            seen.append(q[:,0,0].tolist())
            self.assertTrue(torch.equal(k[64:68,0,0], torch.arange(64,68).float()))
            return torch.ones(len(k))
        def kernel(q,k,v,out,qs,qn,ks,kn,positions,maxq,**kwargs):
            sink_seen.append(positions.tolist());out.zero_()
        def reduce(value):
            reductions.append(value.clone());return 2*value
        def broadcast(value,src):
            broadcasts.append((src, value.clone()));return value
        modules = {'vllm.forward_context':NS(get_forward_context=lambda:context),
                   'vllm.distributed':NS(get_tp_group=lambda:NS(world_size=2,broadcast=broadcast), tensor_model_parallel_all_reduce=reduce)}
        with tempfile.TemporaryDirectory() as root:
            write_f3(root, 'fixture', list(range(64,128)), [.5]*64)
            sparse = NS(method='tree-prophetkv', scoring_layers=tuple(range(64)), ratio=.2,
                model=NS(layers=layers,get_input_embeddings=lambda ids:ids.float()[:,None].expand(-1,2)),
                prompt_token_ids=tuple(range(384)), request=RequestMetadata('r',(0,64,128,384),(200,201)),
                selection_diagnostics=[], attn_metadata=NS(block_table=torch.arange(6)[None]),
                connector=NS(wait_for_layer_load=aligned.add, prophet_aligned=aligned),
                tree_f3_config=dict(cache_dir=root,namespace='fixture',readonly=True))
            with patch.dict('sys.modules', modules), patch.object(runtime,'project',project), patch.object(runtime,'context_importance',importance), patch.object(attention,'positional_kernel',return_value=kernel), patch('torch.cuda.synchronize'):
                selected = runtime.probe(sparse,torch.arange(128,384),torch.zeros(256,2))
        self.assertEqual(seen, [list(map(float,range(200,384)))]*64)
        self.assertEqual(sink_seen, [[64,65,66,67]]*64)
        self.assertEqual(selected.tolist(),list(range(64,76)))
        self.assertEqual(sparse.request.question_positions,(200,201))
        self.assertEqual(len(aligned),64);self.assertEqual(len(reductions),1);self.assertEqual(broadcasts[0][0],0)
        torch.testing.assert_close(cache,before)
        self.assertEqual(sparse.selection_diagnostics[0]['question_positions'],[200,201])

    def test_tree_population_never_prefix_reuses_existing_kv(self):
        from unittest.mock import Mock
        from test_rpkv import connector, request
        from runner.layout import request_metadata
        tokens = list(range(128)); rid = 'a'*32+':not-a-phase-name'
        req = NS(request_id=rid, prompt_token_ids=tokens, all_token_ids=tokens,
            num_prompt_tokens=128, num_tokens=128,
            sampling_params=NS(extra_args={'rpkv_layout':request_metadata(rid,tokens,'populate')}))
        for tree in (False,True):
            c=connector()
            c.launch_config={'tree_f3': {'enabled': True}} if tree else {}
            c.store=NS(lookup=Mock(return_value=[True,True]))
            matched,_=c.get_num_new_matched_tokens(req,0)
            self.assertEqual(matched,0 if tree else 127)
            if tree:
                c.store.lookup.assert_not_called()
                self.assertEqual(c.requests_blend_meta[rid].pc_hit_block_num,0)
            else:
                c.store.lookup.assert_called_once()

    def test_population_uses_validated_phase_and_cleans_cancelled_state(self):
        import tempfile
        from pathlib import Path
        from ucm.sparse.prophetkv.tree_runtime import begin_population, record_layer
        from runner.layout import request_metadata
        from runner.tree_prophetkv import read_f3
        from test_chunked_prefill import WorkerStateTests
        from ucm.sparse.prophetkv.prophetkv import ProphetKV
        tokens = list(range(64));rid='a'*32+':arbitrary-name'
        layout=request_metadata(rid,tokens,'populate')
        with tempfile.TemporaryDirectory() as root:
            sparse=WorkerStateTests.sparse()
            sparse.tree_f3_config=dict(cache_dir=root,namespace='fixture',readonly=False)
            sparse.model_config_hf=NS(num_hidden_layers=64,hidden_size=2,num_attention_heads=1,head_dim=2)
            begin_population(sparse,layout,rid,tokens,0,64)
            q=torch.ones(64,1,2)
            modules={'vllm.distributed':NS(get_tensor_model_parallel_rank=lambda:0,get_tp_group=lambda:NS(world_size=1),tensor_model_parallel_all_reduce=lambda x:x)}
            with patch.dict('sys.modules',modules):
                for i in range(16): record_layer(sparse,i,q,q)
            self.assertIsNone(sparse.f3rec)
            self.assertEqual(len(read_f3(root,'fixture',tokens)),64)
            sparse.request_finished_in_worker(rid)
            self.assertIsNone(sparse.f3_request_id)
            begin_population(sparse,layout,rid,tokens,0,64)
            sparse.request_finished_in_worker(rid)
            self.assertIsNone(sparse.f3rec);self.assertIsNone(sparse.f3_tokens)
            with self.assertRaises(ValueError):begin_population(sparse,None,rid,tokens,0,64)
            sparse.tree_f3_config['readonly']=True
            with self.assertRaises(RuntimeError):begin_population(sparse,layout,rid,tokens,0,64)

    def test_attention_dispatch_masks_contiguous_rows_and_leaves_decode_native(self):
        from unittest.mock import Mock
        from ucm.sparse.prophetkv import attention
        # All context rows selected: still use masked dispatch even though this
        # prefill is contiguous. Decode must always use the original backend.
        for positions in (torch.arange(64,128),torch.arange(128,144)):
            impl=type('FlashAttentionImpl',(),{})()
            native=Mock(side_effect=lambda *args: args[-2]);impl.forward=native
            impl.kv_cache_dtype='auto';impl.kv_sharing_target_layer_name=None
            impl.sliding_window=(-1,-1);impl.logits_soft_cap=0;impl.scale=.5
            layer=NS(impl=impl,_k_scale=1.,_v_scale=1.)
            sparse=NS(active=True,method='tree-prophetkv',audit=True,pending_audit=[{}],connector=NS(),
                fusion_positions=positions,fusion_slots=torch.arange(int(positions[-1])+1),block_size=64,
                request=NS(boundaries=[0,64,128,384]),coverage_selected=torch.arange(64,128))
            q=torch.randn(len(positions),2,4);k=torch.randn(len(positions),1,4);v=torch.randn_like(k)
            kv=torch.randn(2,6,64,1,4);out=torch.empty_like(q)
            def write(k,v,kc,vc,slots,*args):
                kc[slots//64,slots%64]=k;vc[slots//64,slots%64]=v
            def kernel(q,kk,vv,out,qs,qn,ks,kn,pos,maxq,**kwargs):
                for a,n,start,length in zip(qs,qn,ks,kn):
                    out[a:a+n]=MaskTests.attention(q[a:a+n],kk[start:start+length],vv[start:start+length],pos[a:a+n],kwargs['sm_scale'])[0]
            def flash(q,kk,vv,*,causal,softmax_scale,**kwargs):
                p=torch.arange(len(kk)-len(q),len(kk)) if causal else None
                return MaskTests.attention(q,kk,vv,p,softmax_scale)
            modules={'vllm.v1.attention.backends.flash_attn':NS(reshape_and_cache_flash=write),
                     'vllm.vllm_flash_attn':NS(flash_attn_varlen_func=flash)}
            with patch.dict('sys.modules',modules),patch.object(attention,'positional_kernel',return_value=kernel):
                attention.install(layer,sparse)
                impl.forward(layer,q,k,v,kv,NS(use_cascade=False,slot_mapping=positions),out)
                native.assert_not_called()
                self.assertTrue(sparse.pending_audit[-1]['causal_attention_verified'])
                sparse.active=False
                impl.forward(layer,q,k,v,kv,None,out)
                native.assert_called_once()


if __name__ == '__main__': unittest.main()
