import unittest

from new.datasets import DATASET_REGISTRY, get_dataset_spec, load_dataset


class DatasetTests(unittest.TestCase):
    def test_dataset_registry_paths_exist_and_civil_uses_zai_namespace(self):
        civil = get_dataset_spec("civil")
        self.assertEqual(civil.namespace, "zai")
        for spec in DATASET_REGISTRY.values():
            self.assertTrue(spec.neutral_questions_path.is_file())
            for persona_spec in spec.persona_specs.values():
                self.assertTrue(persona_spec.questions_path.is_file())
                self.assertTrue(persona_spec.answers_path.is_file())

    def test_expected_chunk_id_construction_with_limits(self):
        civil = load_dataset("civil", limit_personas={"civil"}, limit_questions=2)
        self.assertEqual(civil.expected_chunk_ids["civil"], ["civil-1", "civil-4"])

        sports = load_dataset("sports2", limit_personas={"football"}, limit_questions=3)
        self.assertEqual(
            sports.expected_chunk_ids["football"],
            ["football-1", "football-1", "football-10"],
        )

    def test_science_dataset_loads_all_personas(self):
        science = load_dataset("science", limit_questions=1)
        self.assertEqual(
            set(science.persona_questions),
            {"biology", "chemistry", "physics"},
        )
        self.assertEqual(science.expected_chunk_ids["biology"], ["biology-1"])


if __name__ == "__main__":
    unittest.main()
