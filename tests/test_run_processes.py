"""Process intervention rejects PID reuse while preserving orphan cleanup."""
import signal
import unittest
from unittest.mock import patch

from runner import operations
from scripts.launcher import launcher_stop


class OwnedProcessTests(unittest.TestCase):
    pid = 987654
    saved = dict(start='100', cmd=b'python\0owned-worker.py\0'.hex())

    def row(self, *, start='100', command=None, session=None):
        return dict(state='S', parent=1, group=self.pid,
                    session=self.pid if session is None else session,
                    identity=dict(start=start, cmd=self.saved['cmd'] if command is None else command))

    def test_reused_pid_or_changed_command_is_rejected_before_any_signal(self):
        for change in ({'start':'101'}, {'command':b'python\0unrelated.py\0'.hex()}):
            with self.subTest(change=change), \
                    patch.object(launcher_stop,'process_table',return_value={self.pid:self.row(**change)}), \
                    patch.object(operations.os,'killpg') as kill:
                with self.assertRaisesRegex(RuntimeError,'identity changed'):
                    operations.terminate_owned(dict(pid=self.pid,identity=self.saved))
                kill.assert_not_called()

    def test_exact_owned_leader_is_signalled_once_then_retired(self):
        with patch.object(launcher_stop,'process_table',side_effect=[{self.pid:self.row()},{}]), \
                patch.object(operations,'group_alive',return_value=False), \
                patch.object(operations.os,'killpg') as kill:
            operations.terminate_owned(dict(pid=self.pid,identity=self.saved))
        kill.assert_called_once_with(self.pid,signal.SIGTERM)

    def test_absent_or_zombie_leader_allows_verified_surviving_ranks(self):
        # process_table excludes zombies, as does identity(); only the orphan
        # child remains. Its session and birth time still establish ownership.
        child=self.pid+1
        with patch.object(launcher_stop,'process_table',side_effect=[{child:self.row(start='105')},{}]), \
                patch.object(operations,'identity',return_value=None), \
                patch.object(operations,'group_alive',return_value=False), \
                patch.object(operations.os,'killpg') as kill:
            operations.terminate_owned(dict(pid=self.pid,identity=self.saved))
        kill.assert_called_once_with(self.pid,signal.SIGTERM)

    def test_orphan_with_wrong_session_or_older_start_is_rejected(self):
        for change in ({'session':self.pid-1}, {'start':'99'}):
            with self.subTest(change=change), \
                    patch.object(launcher_stop,'process_table',return_value={self.pid+1:self.row(**change)}), \
                    patch.object(operations.os,'killpg') as kill:
                with self.assertRaisesRegex(RuntimeError,'ownership changed'):
                    operations.terminate_owned(dict(pid=self.pid,identity=self.saved))
                kill.assert_not_called()


if __name__=='__main__':unittest.main()
