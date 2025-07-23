"""Core definitions of mrgdvdb calculations documents."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from emmet.core.structure import StructureMetadata
from jobflow.utils import ValueEnum
from pydantic import Field

from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.utils.common import get_mrgdv_report
from atomate2.utils.path import get_uri

logger = logging.getLogger(__name__)


class TaskState(ValueEnum):
    """Mrgdv calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"
    UNCONVERGED = "unconverged"


class MrgdvdbObject(ValueEnum):
    """Types of Mrgdvdb data objects."""

    DVDBFILE = "out_DVDB"  # DVDB file as string


class MrgdvdbTaskDoc(StructureMetadata):
    """Definition of task document about an Mrgdv Job.

    Parameters
    ----------
    dir_name: str
        The directory for this Abinit task
    completed_at: str
        Timestamp for when this task was completed
    state: .TaskState
        State of this task
    included_objects: List[.MrgdvdbObject]
        List of Abinit objects included with this task document
    mrgdv_objects: Dict[.MrgdvdbObject, Any]
        Abinit objects associated with this task
    task_label: str
        A description of the task
    tags: List[str]
        Metadata tags for this task document
    """

    dir_name: Optional[str] = Field(
        None, description="The directory for this Abinit task"
    )
    completed_at: Optional[str] = Field(
        None, description="Timestamp for when this task was completed"
    )
    included_objects: Optional[list[MrgdvdbObject]] = Field(
        None, description="List of Mrgdv objects included with this task document"
    )
    mrgdv_objects: Optional[dict[MrgdvdbObject, Any]] = Field(
        None, description="Mrgdv objects associated with this task"
    )
    task_label: Optional[str] = Field(None, description="A description of the task")

    tags: Optional[list[str]] = Field(
        None, description="Metadata tags for this task document"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        additional_fields: dict[str, Any] = None,
    ) -> MrgdvdbTaskDoc:
        """Create a task document from a directory containing Abinit/Mrgdv files.

        Parameters
        ----------
        dir_name: Path or str
            The path to the folder containing the calculation outputs.
        additional_fields: Dict[str, Any]
            Dictionary of additional fields to add to output document.

        Returns
        -------
        .MrgdvdbTaskDoc
            A task document for the calculation.
        """
        logger.info(f"Getting task doc in: {dir_name}")

        if additional_fields is None:
            additional_fields = {}

        dir_name = Path(dir_name)
        task_files = _find_abinit_files(dir_name)

        if len(task_files) == 0:
            raise FileNotFoundError("No Abinit files found!")
        if len(task_files) > 1:
            raise RuntimeError(
                f"Only one mrgdv calculation expected. Found {len(task_files)}"
            )

        std_task_files = next(iter(task_files.values()))
        abinit_mrgdvdb_file = std_task_files["abinit_outdvdb_file"]

        if not abinit_mrgdvdb_file.exists():
            raise RuntimeError(
                f"The output DVDB file {abinit_mrgdvdb_file} does not exist"
            )

        mrgdv_objects: dict[MrgdvdbObject, Any] = {}
        mrgdv_objects[MrgdvdbObject.DVDBFILE] = AbinitStoredFile.from_file(  # type: ignore[index]
            filepath=abinit_mrgdvdb_file,
            data_type=bytes,
        )

        completed_at = str(
            datetime.fromtimestamp(
                os.stat(abinit_mrgdvdb_file).st_mtime, tz=timezone.utc
            )
        )

        report = get_mrgdv_report(logfile=std_task_files["abinit_mrglog_file"])

        if not report["run_completed"]:
            raise RuntimeError("mrgdv execution was not completed")

        tags = additional_fields.get("tags")

        dir_name = get_uri(dir_name)  # convert to full uri path

        included_objects = None
        if mrgdv_objects:
            included_objects = list(mrgdv_objects)

        data = {
            "dir_name": dir_name,
            "completed_at": completed_at,
            "included_objects": included_objects,
            "mrgdv_objects": mrgdv_objects,
            "tags": tags,
        }

        doc = cls(**data)
        return doc.model_copy(update=additional_fields, deep=True)


def _find_abinit_files(
    path: Path | str,
) -> dict[str, Any]:
    """Find Abinit files."""
    path = Path(path)
    task_files = {}

    def _get_task_files(files: list[Path], suffix: str = "") -> dict:
        abinit_files = {}
        for file in files:
            # Here we make assumptions about the output file naming
            if file.match(f"*outdata/out_dv{suffix}*"):
                abinit_files["abinit_outdvdb_file"] = Path(file).relative_to(path)
            elif file.match(f"*run.log{suffix}*"):
                abinit_files["abinit_mrglog_file"] = Path(file).relative_to(path)

        return abinit_files

    # get any matching file from the root folder
    standard_files = _get_task_files(
        list(path.glob("*")) + list(path.glob("outdata/*"))
    )
    if len(standard_files) > 0:
        task_files["standard"] = standard_files

    return task_files
