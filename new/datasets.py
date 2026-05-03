"""Dataset registry and local file loading for standalone experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PersonaSpec:
    """Local files and chunk ID mapping for one persona."""

    name: str
    questions_path: Path
    answers_path: Path
    chunk_id_prefix: str


@dataclass(frozen=True)
class DatasetSpec:
    """Static dataset configuration."""

    name: str
    namespace: str
    neutral_questions_path: Path
    personas: tuple[str, ...]
    persona_specs: dict[str, PersonaSpec]


@dataclass(frozen=True)
class LoadedDataset:
    """Dataset contents loaded from disk."""

    spec: DatasetSpec
    neutral_questions: list[str]
    persona_questions: dict[str, list[str]]
    expected_chunk_ids: dict[str, list[str]]


def _path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "civil": DatasetSpec(
        name="civil",
        namespace="zai",
        neutral_questions_path=_path("data", "zai", "questions", "both.txt"),
        personas=("civil", "minecraft"),
        persona_specs={
            "civil": PersonaSpec(
                name="civil",
                questions_path=_path("data", "zai", "questions", "civil.txt"),
                answers_path=_path("data", "zai", "answers", "civil.txt"),
                chunk_id_prefix="civil",
            ),
            "minecraft": PersonaSpec(
                name="minecraft",
                questions_path=_path("data", "zai", "questions", "minecraft.txt"),
                answers_path=_path("data", "zai", "answers", "minecraft.txt"),
                chunk_id_prefix="minecraft",
            ),
        },
    ),
    "science": DatasetSpec(
        name="science",
        namespace="science",
        neutral_questions_path=_path("data", "science", "questions", "both.txt"),
        personas=("biology", "chemistry", "physics"),
        persona_specs={
            "biology": PersonaSpec(
                name="biology",
                questions_path=_path("data", "science", "questions", "biology.txt"),
                answers_path=_path("data", "science", "answers", "biology.txt"),
                chunk_id_prefix="biology",
            ),
            "chemistry": PersonaSpec(
                name="chemistry",
                questions_path=_path("data", "science", "questions", "chemistry.txt"),
                answers_path=_path("data", "science", "answers", "chemistry.txt"),
                chunk_id_prefix="chemistry",
            ),
            "physics": PersonaSpec(
                name="physics",
                questions_path=_path("data", "science", "questions", "physics.txt"),
                answers_path=_path("data", "science", "answers", "physics.txt"),
                chunk_id_prefix="physics",
            ),
        },
    ),
    "sports2": DatasetSpec(
        name="sports2",
        namespace="sports2",
        neutral_questions_path=_path("data", "sports2", "questions", "all.txt"),
        personas=("football", "basketball", "soccer", "hockey"),
        persona_specs={
            "football": PersonaSpec(
                name="football",
                questions_path=_path("data", "sports2", "questions", "football.txt"),
                answers_path=_path("data", "sports2", "answers", "football.txt"),
                chunk_id_prefix="football",
            ),
            "basketball": PersonaSpec(
                name="basketball",
                questions_path=_path("data", "sports2", "questions", "basketball.txt"),
                answers_path=_path("data", "sports2", "answers", "basketball.txt"),
                chunk_id_prefix="basketball",
            ),
            "soccer": PersonaSpec(
                name="soccer",
                questions_path=_path("data", "sports2", "questions", "soccer.txt"),
                answers_path=_path("data", "sports2", "answers", "soccer.txt"),
                chunk_id_prefix="soccer",
            ),
            "hockey": PersonaSpec(
                name="hockey",
                questions_path=_path("data", "sports2", "questions", "hockey.txt"),
                answers_path=_path("data", "sports2", "answers", "hockey.txt"),
                chunk_id_prefix="hockey",
            ),
        },
    ),
}


def dataset_names() -> tuple[str, ...]:
    """Return supported dataset names in default execution order."""
    return ("civil", "science", "sports2")


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return one dataset spec by name."""
    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(dataset_names())
        raise ValueError(f"unknown dataset '{name}'. Supported datasets: {supported}") from exc


def load_questions(path: Path) -> list[str]:
    """Load non-empty lines from a text file."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Question file not found: {resolved}")
    questions = [
        line.strip()
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise RuntimeError(f"No questions found in {resolved}")
    return questions


def load_expected_chunk_ids(
    path: Path,
    persona: str,
    expected_count: int,
    *,
    chunk_id_prefix: str,
) -> list[str]:
    """Load answer numbers and convert them to `<file-stem>-<number>` chunk IDs."""
    answers = load_questions(path)
    if len(answers) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} answers for {persona}, found {len(answers)} "
            f"in {path.expanduser().resolve()}"
        )
    chunk_ids: list[str] = []
    for line_number, answer in enumerate(answers, start=1):
        if not answer.isdigit():
            raise RuntimeError(
                f"Answer line {line_number} for {persona} must be a chunk number: "
                f"{answer!r}"
            )
        chunk_ids.append(f"{chunk_id_prefix}-{int(answer)}")
    return chunk_ids


def load_dataset(
    name: str,
    *,
    limit_personas: set[str] | None = None,
    limit_questions: int | None = None,
) -> LoadedDataset:
    """Load a configured dataset from disk."""
    spec = get_dataset_spec(name)
    neutral_questions = load_questions(spec.neutral_questions_path)
    full_question_count = len(neutral_questions)
    if limit_questions is not None:
        if limit_questions < 1:
            raise ValueError("limit_questions must be at least 1")
        neutral_questions = neutral_questions[:limit_questions]

    selected_personas = [
        persona
        for persona in spec.personas
        if limit_personas is None or persona in limit_personas
    ]
    persona_questions: dict[str, list[str]] = {}
    expected_chunk_ids: dict[str, list[str]] = {}
    for persona in selected_personas:
        persona_spec = spec.persona_specs[persona]
        persona_questions[persona] = load_questions(persona_spec.questions_path)
        expected = load_expected_chunk_ids(
            persona_spec.answers_path,
            persona,
            full_question_count,
            chunk_id_prefix=persona_spec.chunk_id_prefix,
        )
        expected_chunk_ids[persona] = expected[: len(neutral_questions)]

    return LoadedDataset(
        spec=spec,
        neutral_questions=neutral_questions,
        persona_questions=persona_questions,
        expected_chunk_ids=expected_chunk_ids,
    )

