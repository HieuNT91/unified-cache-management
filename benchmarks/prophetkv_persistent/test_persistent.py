import unittest
from lifecycle import TrackedStore, seed_value, namespace_from_id

class Store:
    def __init__(self): self.waited=[]; self.fail=False
    def load_data(self): return object()
    dump_data=load_data
    def wait(self,task):
        if self.fail: raise RuntimeError('backend failure')
        self.waited.append(task)

class LifecycleTests(unittest.TestCase):
    def test_distinct_namespaces(self):
        self.assertNotEqual(seed_value('a'*32), seed_value('b'*32))
        self.assertEqual(namespace_from_id('a'*32+':measured'), 'a'*32)
        for bad in ('measured','../oops:x','a'*32+':'):
            with self.assertRaises(ValueError): namespace_from_id(bad)
    def test_all_operations_drained(self):
        backend=Store(); store=TrackedStore(backend)
        a=store.load_data(); b=store.dump_data()
        store.wait(a)
        self.assertEqual(store.drain(),dict(pending=0,completed=2))
        self.assertEqual(backend.waited,[a,b])
    def test_failed_wait_not_retired(self):
        backend=Store(); store=TrackedStore(backend); store.dump_data(); backend.fail=True
        with self.assertRaises(RuntimeError):store.drain()
        self.assertEqual(len(store.pending),1)
        backend.fail=False
        self.assertEqual(store.drain()['pending'],0)

class RetirementTests(unittest.TestCase):
    def test_backend_directories_survive_and_unknown_files_fail_closed(self):
        import tempfile
        from pathlib import Path
        from lifecycle import delete_retired_files
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'kv/.temp').mkdir(parents=True)
            path=root/'kv/aaaaaaaa'/('a'*32);path.parent.mkdir();path.write_bytes(b'kv')
            with self.assertRaises(RuntimeError):delete_retired_files(root, [])
            self.assertTrue(path.exists())
            delete_retired_files(root, [str(path.relative_to(root))])
            self.assertTrue((root/'kv/.temp').is_dir())
            self.assertFalse(path.exists())

if __name__=='__main__': unittest.main()
