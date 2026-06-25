"""
Curriculum learning strategies for ASR fine-tuning.

Implements progressive training from simple to complex utterances based on
the Moonshine paper recommendations (arXiv:2410.15608v2).

Key findings from paper:
- Optimal training instance duration: [4, 30] seconds
- Bimodal distribution emerges naturally from preprocessing
- <0.5% of training data should be <1 second (causes repetitions, >100% WER)
- Model trained on variable-length sequences (no padding)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import datasets


@dataclass
class CurriculumPhase:
    """
    Configuration for a single curriculum learning phase.

    Attributes:
        name: Human-readable phase name
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
        max_words: Maximum number of words (None = unlimited)
        max_steps: Training steps for this phase
        learning_rate: Learning rate for this phase
        warmup_steps: Number of warmup steps
        label_smoothing: Label smoothing factor
        repetition_penalty: Penalty for repetitive outputs
        num_beams: Number of beams for beam search
        no_repeat_ngram_size: Size of n-grams that cannot repeat
        target_wer: Target WER percentage for this phase
    """
    name: str
    min_duration: float
    max_duration: float
    max_words: Optional[int]
    max_steps: int
    learning_rate: float
    warmup_steps: int
    label_smoothing: float = 0.1
    repetition_penalty: float = 1.2
    num_beams: int = 5
    no_repeat_ngram_size: int = 2
    target_wer: float = 25.0
    description: str = ""


# Default 3-phase curriculum following Moonshine paper recommendations
# Paper: Training instances in [4, 30] seconds with bimodal distribution
# Paper: <0.5% of data should be <1 second (causes repetitions)
DEFAULT_CURRICULUM = [
    CurriculumPhase(
        name="Phase 1: Short Utterances (4-10s)",
        description="Foundation training on shorter, clearer utterances",
        min_duration=4.0,  # Paper minimum: 4 seconds
        max_duration=10.0,
        max_words=15,  # Roughly 1.5 words/second
        max_steps=4000,
        learning_rate=1e-5,
        warmup_steps=500,
        label_smoothing=0.1,
        repetition_penalty=1.5,  # Higher penalty for short utterances
        num_beams=3,
        no_repeat_ngram_size=3,
        target_wer=20.0
    ),
    CurriculumPhase(
        name="Phase 2: Medium Utterances (10-20s)",
        description="Sequence learning on medium-length utterances",
        min_duration=10.0,
        max_duration=20.0,
        max_words=30,  # Roughly 1.5 words/second
        max_steps=6000,
        learning_rate=3e-5,
        warmup_steps=800,
        label_smoothing=0.1,
        repetition_penalty=1.2,
        num_beams=5,
        no_repeat_ngram_size=2,
        target_wer=30.0
    ),
    CurriculumPhase(
        name="Phase 3: Full Range (4-30s, Bimodal)",
        description="Full complexity with bimodal distribution as per paper",
        min_duration=4.0,  # Paper: [4, 30] seconds
        max_duration=30.0,
        max_words=None,  # No limit
        max_steps=5000,
        learning_rate=5e-6,
        warmup_steps=400,
        label_smoothing=0.05,
        repetition_penalty=1.0,  # Allow natural speech patterns
        num_beams=5,
        no_repeat_ngram_size=0,  # Disable to allow valid repetitions
        target_wer=25.0
    )
]


class CurriculumScheduler:
    """
    Manages curriculum learning progression for ASR fine-tuning.

    Filters datasets by duration ranges based on Moonshine paper recommendations.
    The paper trained on [4, 30] second instances with a bimodal distribution.
    """

    def __init__(self, phases: Optional[List[CurriculumPhase]] = None):
        """
        Initialize curriculum scheduler.

        Args:
            phases: List of curriculum phases (uses default if None)
        """
        self.phases = phases or DEFAULT_CURRICULUM
        self.current_phase_idx = 0

    def get_phase(self, phase_idx: int) -> CurriculumPhase:
        """Get phase configuration by index."""
        if phase_idx < 0 or phase_idx >= len(self.phases):
            raise ValueError(f"Invalid phase index: {phase_idx}. Must be 0-{len(self.phases)-1}")
        return self.phases[phase_idx]

    def get_current_phase(self) -> CurriculumPhase:
        """Get current phase configuration."""
        return self.phases[self.current_phase_idx]

    def next_phase(self) -> Optional[CurriculumPhase]:
        """
        Move to next phase.

        Returns:
            Next phase configuration, or None if already at last phase
        """
        if self.current_phase_idx < len(self.phases) - 1:
            self.current_phase_idx += 1
            return self.get_current_phase()
        return None

    def filter_dataset(
        self,
        dataset: datasets.Dataset,
        phase: CurriculumPhase,
        duration_column: str = "duration",
        text_column: str = "sentence"
    ) -> datasets.Dataset:
        """
        Filter dataset based on phase criteria (duration and word count).

        Following Moonshine paper recommendations:
        - Training instances in [4, 30] seconds
        - <0.5% of data should be <1 second (causes repetitions)

        Args:
            dataset: Input dataset
            phase: Curriculum phase configuration
            duration_column: Name of column containing audio duration
            text_column: Name of column containing transcription text

        Returns:
            Filtered dataset
        """
        def meets_criteria(duration, text, audio):
            # Resolve duration if None by calculating from audio array
            if duration is None and audio is not None:
                if 'array' in audio and audio['array'] is not None:
                    duration = len(audio['array']) / audio['sampling_rate']

            if duration is None:
                print(f"[DEBUG] duration is None!")
                return False

            word_count = len(text.split()) if text else 0
            meets = (phase.min_duration <= duration <= phase.max_duration) and (phase.max_words is None or word_count <= phase.max_words)
            if meets:
                print(f"[DEBUG] MEETS: dur={duration:.2f}s, words={word_count}")
            return meets

        # Count samples <1 second for warning (paper recommendation)
        total_count = len(dataset)
        durations_all = [d if d is not None else 5.0 for d in dataset[duration_column]]
        very_short = sum(1 for d in durations_all if d < 1.0)
        very_short_pct = 100 * very_short / total_count if total_count > 0 else 0

        # Use input_columns to avoid decoding audio
        filtered = dataset.filter(meets_criteria, input_columns=[duration_column, text_column, "audio"])

        print(f"\n{phase.name}:")
        print(f"  Duration range: [{phase.min_duration}s, {phase.max_duration}s]")
        print(f"  Max words: {phase.max_words or 'unlimited'}")
        print(f"  Original samples: {total_count:,}")
        print(f"  Filtered samples: {len(filtered):,}")
        print(f"  Retention: {100 * len(filtered) / total_count:.1f}%")

        # Warn about very short audio (paper finding)
        if very_short > 0:
            if very_short_pct < 0.5:
                print(f"  [OK] Very short audio (<1s): {very_short} ({very_short_pct:.2f}%) - within paper recommendation (<0.5%)")
            else:
                print(f"  [WARNING] Very short audio (<1s): {very_short} ({very_short_pct:.2f}%) - exceeds paper recommendation (<0.5%)")
                print(f"    Paper finding: <1s audio causes repetitions and >100% WER")

        # Calculate duration statistics for filtered dataset
        if len(filtered) > 0:
            durations = [
                d if d is not None else len(a['array']) / a['sampling_rate']
                for d, a in zip(filtered[duration_column], filtered["audio"])
            ]
            avg_duration = sum(durations) / len(durations)
            min_dur = min(durations)
            max_dur = max(durations)
            print(f"  Duration stats: min={min_dur:.1f}s, max={max_dur:.1f}s, avg={avg_duration:.1f}s")

        return filtered

    def get_training_args(self, phase: CurriculumPhase, base_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get training arguments for a phase.

        Args:
            phase: Curriculum phase configuration
            base_args: Base training arguments

        Returns:
            Updated training arguments
        """
        args = base_args.copy()
        args.update({
            'learning_rate': phase.learning_rate,
            'max_steps': phase.max_steps,
            'warmup_steps': phase.warmup_steps,
            'label_smoothing_factor': phase.label_smoothing,
        })
        return args

    def get_generation_config(self, phase: CurriculumPhase) -> Dict[str, Any]:
        """
        Get generation configuration for a phase.

        Args:
            phase: Curriculum phase configuration

        Returns:
            Generation configuration dictionary
        """
        return {
            'repetition_penalty': phase.repetition_penalty,
            'num_beams': phase.num_beams,
            'no_repeat_ngram_size': phase.no_repeat_ngram_size,
            'length_penalty': 1.2,
            'do_sample': False,
            'early_stopping': True,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CurriculumScheduler':
        """
        Create curriculum scheduler from configuration dictionary.

        Args:
            config: Configuration dictionary with 'curriculum' key

        Returns:
            Initialized CurriculumScheduler
        """
        curriculum_config = config.get('curriculum', {})

        if not curriculum_config.get('enabled', True):
            # If curriculum is disabled, create single phase with full data
            # Use paper's recommended range [4, 30] seconds
            phases = [CurriculumPhase(
                name="Full Dataset Training (4-30s)",
                description="Training on complete dataset without curriculum (paper range)",
                min_duration=4.0,  # Paper minimum
                max_duration=30.0,  # Paper maximum
                max_words=None,
                max_steps=config.get('training', {}).get('max_steps', 15000),
                learning_rate=config.get('training', {}).get('learning_rate', 1e-5),
                warmup_steps=config.get('training', {}).get('warmup_steps', 500),
            )]
            return cls(phases)

        # Parse phases from config
        phases = []
        for phase_config in curriculum_config.get('phases', []):
            phase = CurriculumPhase(
                name=phase_config.get('name', 'Unnamed Phase'),
                description=phase_config.get('description', ''),
                min_duration=phase_config.get('min_duration', 4.0),
                max_duration=phase_config.get('max_duration', 30.0),
                max_words=phase_config.get('max_words'),
                max_steps=phase_config.get('max_steps', 5000),
                learning_rate=phase_config.get('learning_rate', 1e-5),
                warmup_steps=phase_config.get('warmup_steps', 500),
                label_smoothing=phase_config.get('label_smoothing', 0.1),
                repetition_penalty=phase_config.get('repetition_penalty', 1.2),
                num_beams=phase_config.get('num_beams', 5),
                no_repeat_ngram_size=phase_config.get('no_repeat_ngram_size', 2),
                target_wer=phase_config.get('target_wer', 25.0),
            )
            phases.append(phase)

        if not phases:
            # Use default curriculum if no phases specified
            phases = DEFAULT_CURRICULUM

        return cls(phases)

    def print_summary(self):
        """Print summary of curriculum learning strategy."""
        print("\n" + "="*80)
        print("CURRICULUM LEARNING STRATEGY")
        print("="*80)
        print(f"Based on Moonshine paper (arXiv:2410.15608v2) recommendations:")
        print(f"  - Training instance duration: [4, 30] seconds")
        print(f"  - Bimodal distribution (naturally emerges)")
        print(f"  - <0.5% of data should be <1s (avoids repetition problems)")
        print(f"\nNumber of phases: {len(self.phases)}")
        print("="*80)

        for i, phase in enumerate(self.phases, 1):
            print(f"\n{phase.name}")
            print(f"  Description: {phase.description}")
            print(f"  Duration: [{phase.min_duration}s, {phase.max_duration}s]")
            print(f"  Max words: {phase.max_words or 'unlimited'}")
            print(f"  Steps: {phase.max_steps:,}")
            print(f"  Learning rate: {phase.learning_rate}")
            print(f"  Target WER: <{phase.target_wer}%")

        print("="*80 + "\n")
