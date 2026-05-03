import csv
import tempfile
import unittest
from pathlib import Path

from new.evaluator import EvaluationResult, RetrievalRun
from new.experiment_config import AlphaConfig
from new.reporting import summarize_results, write_detailed_csv, write_summary_csv


def _fake_result() -> EvaluationResult:
    return EvaluationResult(
        run_id="run-test",
        dataset="civil",
        namespace="zai",
        index_name="541",
        user_namespace="users",
        embed_model="llama-text-embed-v2",
        top_k=5,
        combine_strategy="linear-comb",
        rerank_strategy="none",
        update_strategy="moving-average",
        alpha_config=AlphaConfig(mode="static", value=0.8),
        final_alpha=0.8,
        persona="civil",
        question_number=1,
        username="user",
        neutral_question="question",
        expected_chunk_id="civil-1",
        priming_question_count=2,
        baseline=RetrievalRun(["minecraft-1"], 0.0, None),
        post_priming=RetrievalRun(["civil-1"], 10.0, 1),
    )


class ReportingTests(unittest.TestCase):
    def test_csv_writers_with_fake_results(self):
        result = _fake_result()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            details_path = write_detailed_csv([result], tmp_path / "details.csv")
            summary_path = write_summary_csv([result], tmp_path / "summary.csv")

            with details_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(rows[0]["namespace"], "zai")
            self.assertEqual(rows[0]["delta"], "10.000000")

            with summary_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(rows[0]["case_count"], "1")
            self.assertEqual(rows[0]["expected_retrieval_rate"], "1.000000")

    def test_summarize_results_groups_by_experiment_factors(self):
        rows = summarize_results([_fake_result()])
        self.assertEqual(rows[0]["mean_post_score"], "10.000000")
        self.assertEqual(rows[0]["wins"], 1)


if __name__ == "__main__":
    unittest.main()
