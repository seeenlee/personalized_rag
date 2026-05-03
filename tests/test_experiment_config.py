import unittest

from new.experiment_config import (
    AlphaConfig,
    alpha_configs_for_strategy,
    alpha_run,
    parse_args,
)


class ExperimentConfigTests(unittest.TestCase):
    def test_static_alpha_run_repeats_value(self):
        run = alpha_run(AlphaConfig(mode="static", value=0.8), priming_count=3)
        self.assertEqual(run.priming_alphas, [0.8, 0.8, 0.8])
        self.assertEqual(run.final_alpha, 0.8)

    def test_sliding_alpha_run_clamps_to_floor(self):
        run = alpha_run(
            AlphaConfig(mode="sliding", start=0.99, step=0.2, floor=0.7),
            priming_count=3,
        )
        self.assertEqual(run.priming_alphas, [0.99, 0.79, 0.7])
        self.assertEqual(run.final_alpha, 0.7)

    def test_query_only_has_no_alpha_configs(self):
        configs = alpha_configs_for_strategy("query-only")
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].mode, "none")

    def test_parse_args_defaults_output_paths_include_run_id(self):
        config = parse_args(["--dry-run", "--run-id", "test-run"])
        self.assertEqual(
            config.details_csv.as_posix(),
            "new/results/test-run_details.csv",
        )
        self.assertEqual(
            config.summary_csv.as_posix(),
            "new/results/test-run_summary.csv",
        )


if __name__ == "__main__":
    unittest.main()
