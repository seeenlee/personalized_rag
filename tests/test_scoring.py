import math
import unittest

from new.scoring import find_expected_rank, persona_rank_score


class ScoringTests(unittest.TestCase):
    def test_persona_rank_score_rewards_expected_and_aligned_chunks(self):
        score = persona_rank_score(
            "civil",
            "civil-1",
            ["civil-1", "civil-2", "minecraft-1"],
        )
        self.assertTrue(math.isclose(score, 10.0 + (1.0 / math.log(3, 2))))

    def test_find_expected_rank_returns_none_when_missing(self):
        self.assertEqual(find_expected_rank(["a-1", "b-2"], "b-2"), 2)
        self.assertIsNone(find_expected_rank(["a-1", "b-2"], "c-3"))


if __name__ == "__main__":
    unittest.main()
