import unittest

from utils.plot_executor import execute_plot_code_with_details


class PlotExecutorTest(unittest.TestCase):
    def test_execute_plot_code_with_details_success(self):
        result = execute_plot_code_with_details(
            "import matplotlib.pyplot as plt\nplt.plot([1, 2], [3, 4])"
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["figure_detected"])
        self.assertIsNotNone(result["base64_jpg"])

    def test_execute_plot_code_with_details_captures_exception(self):
        result = execute_plot_code_with_details("raise ValueError('boom')")

        self.assertFalse(result["success"])
        self.assertFalse(result["figure_detected"])
        self.assertIn("ValueError: boom", result["exception"])


if __name__ == "__main__":
    unittest.main()
