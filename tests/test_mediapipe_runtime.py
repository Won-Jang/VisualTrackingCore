"""Check platform startup policy without starting a native graph or a camera."""

import unittest
from unittest.mock import patch

import mediapipe as mp

from adapters.mediapipe_adapter import MediaPipeFaceAdapter


class MediaPipeRuntimeTests(unittest.TestCase):
    def test_macos_unsupported_versions_fail_before_download_or_native_graph(self):
        for version in ("1.0.0", "1.0.1", "unknown"):
            with self.subTest(version=version), \
                    patch("adapters.mediapipe_adapter.platform.system", return_value="Darwin"), \
                    patch.object(mp, "__version__", version), \
                    patch.object(MediaPipeFaceAdapter, "_ensure_model") as download, \
                    patch.object(mp.tasks.vision.FaceLandmarker, "create_from_options") as create:
                with self.assertRaisesRegex(RuntimeError, "python3.11 -m venv .venv-mac"):
                    MediaPipeFaceAdapter()
                download.assert_not_called()
                create.assert_not_called()

    def test_supported_mac_and_windows_select_cpu_and_keep_video_mode(self):
        for system, version in (("Darwin", "0.10.21"), ("Windows", "1.0.1")):
            with self.subTest(system=system), \
                    patch("adapters.mediapipe_adapter.platform.system", return_value=system), \
                    patch.object(mp, "__version__", version), \
                    patch.object(MediaPipeFaceAdapter, "_ensure_model"), \
                    patch.object(mp.tasks.vision.FaceLandmarker, "create_from_options") as create:
                adapter = MediaPipeFaceAdapter(model_path="test-model.task")
                options = create.call_args.args[0]
                self.assertEqual(options.base_options.delegate, mp.tasks.BaseOptions.Delegate.CPU)
                self.assertEqual(options.running_mode, mp.tasks.vision.RunningMode.VIDEO)
                self.assertEqual(options.num_faces, 1)
                adapter.close()
                adapter.close()
                create.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
