from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from robotwin_critic.two_stage_rft.multinode_worker import (
    exported_environment,
    run_job,
    submit_job,
)


class MultinodeWorkerTest(unittest.TestCase):
    def test_exported_environment_does_not_persist_credentials(self) -> None:
        environment = {
            "VISIBLE_SETTING": "yes",
            "SWANLAB_API_KEY": "private",
            "ACCESS_TOKEN": "private",
            "DB_PASSWORD": "private",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            exported = exported_environment()
        self.assertEqual(exported, {"VISIBLE_SETTING": "yes"})

    def test_run_job_releases_holder_lease_after_command(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue"
            control = root / "pause_until"
            output = root / "result.txt"
            submit_job(
                queue,
                "smoke",
                [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
                ],
                root,
            )
            job = json.loads((queue / "jobs" / "smoke.json").read_text())
            result = run_job(
                queue,
                job,
                control,
                lease_seconds=2,
                refresh_seconds=1,
                settle_seconds=0,
            )
            self.assertEqual(result["return_code"], 0)
            self.assertEqual(output.read_text(), "ok")
            self.assertEqual(control.read_text(), "0\n")

    def test_stop_file_terminates_complete_job_session(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue"
            control = root / "pause_until"
            submit_job(
                queue,
                "cancel-smoke",
                [sys.executable, "-c", "import time; time.sleep(30)"],
                root,
            )
            job = json.loads((queue / "jobs" / "cancel-smoke.json").read_text())

            def request_stop() -> None:
                time.sleep(0.2)
                (queue / "STOP").touch()

            thread = threading.Thread(target=request_stop)
            thread.start()
            result = run_job(
                queue,
                job,
                control,
                lease_seconds=2,
                refresh_seconds=1,
                settle_seconds=0,
            )
            thread.join()
            self.assertTrue(result["cancelled"])
            self.assertLess(result["return_code"], 0)
            self.assertEqual(control.read_text(), "0\n")


if __name__ == "__main__":
    unittest.main()
