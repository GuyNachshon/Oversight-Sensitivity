"""
Visualization Artifact

Publication-ready plot with archived source data for reproducibility.
Per data-model.md Entity 6: VisualizationArtifact
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from pathlib import Path
import json


FigureType = Literal["radar", "heatmap", "timeseries", "baseline_comparison", "intervention"]


@dataclass
class VisualizationArtifact:
    """
    Publication-ready plot with source data tracking.

    Per data-model.md validation rules:
    - PNG must be 300 DPI
    - PDF must be vector format
    - Source data must be valid JSON
    - generation_script must be executable Python file
    """

    artifact_id: str
    experiment_id: str
    figure_type: FigureType
    file_path_png: str
    file_path_pdf: str
    source_data_path: str
    generation_script: str
    created_at: datetime

    def __post_init__(self):
        """Validate artifact paths."""
        if self.created_at is None:
            self.created_at = datetime.now()

    @classmethod
    def create(
        cls,
        experiment_id: str,
        figure_type: FigureType,
        output_dir: Path,
        generation_script: str,
    ) -> "VisualizationArtifact":
        """
        Create artifact with standard naming convention.

        Naming: fig_{type}_{experiment_id}.{ext}
        """
        artifact_id = f"fig_{figure_type}_{experiment_id}"

        return cls(
            artifact_id=artifact_id,
            experiment_id=experiment_id,
            figure_type=figure_type,
            file_path_png=str(output_dir / f"{artifact_id}.png"),
            file_path_pdf=str(output_dir / f"{artifact_id}.pdf"),
            source_data_path=str(output_dir / f"{artifact_id}_data.json"),
            generation_script=generation_script,
            created_at=datetime.now(),
        )

    def save_source_data(self, data: dict) -> None:
        """
        Save source data to JSON file for reproducibility.

        Per constitution: Source data must be archived alongside figures.
        """
        path = Path(self.source_data_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(
                {
                    "artifact_id": self.artifact_id,
                    "experiment_id": self.experiment_id,
                    "figure_type": self.figure_type,
                    "created_at": self.created_at.isoformat(),
                    "data": data,
                },
                f,
                indent=2,
            )

    def verify_files_exist(self) -> bool:
        """Validate that all artifact files exist."""
        return all(
            Path(p).exists()
            for p in [self.file_path_png, self.file_path_pdf, self.source_data_path]
        )
