"""Core definitions of mrgdvdb calculations documents."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from emmet.core.structure import StructureMetadata
from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field

from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.utils.common import get_mrgdv_report
from atomate2.utils.path import get_uri

logger = logging.getLogger(__name__)


class TaskState(ValueEnum):
    """Mrgdv calculation state."""

    SUCCESS = "successful"
    FAILED = "failed"
    UNCONVERGED = "unconverged"


# # need to inherit from MSONable to be stored in the data store
# # I tried to combine it with @dataclass, but didn't work...
# class dvFileStr(MSONable):
#     """Object storing the raw string of a dv file."""

#     def __init__(self, dvfilepath: str | Path, dv_as_str: str) -> None:
#         self.dvfilepath: str | Path = dvfilepath
#         self.dv_as_str: str = dv_as_str

#     @classmethod
#     def from_dvfile(cls, dvfile: dvFile) -> Self:
#         """Create a dvFileStr object from the native dvFile abipy object."""
#         with open(dvfile.filepath) as f:
#             dv_as_str = f.read()
#         return cls(dvfilepath=dvfile.filepath, dv_as_str=dv_as_str)


class MrgdvdbObject(ValueEnum):
    """Types of Mrgdvdb data objects."""

    DVDBFILE = "out_DVDB"  # DVDB file as string


class CalculationOutput(BaseModel):
    """Document defining Mrgdv calculation outputs.

    Parameters
    ----------
    dvdb_version: int
        The DVDB file version
    v1scf_number: int
        Number of V1SCF potential
    qpt_number: int
        Number of q-points
    have_dielectric: bool
        Whether the dielectric tensor is present
    have_bec: bool
        Whether the Born effective charges are present
    have_quadrupoles: bool
        Whether the quadrupoles are present
    have_efield: bool
        Whether the derivative wrt the electric field is present
    add_lr: int
        Treatment of long-range part of V1scf
    qdam: float
        Damping factor for Gaussian Filter
    """

    dvdb_version: int = Field(None, description="The DVDB file version")
    v1scf_number: int = Field(None, description="Number of V1SCF potential")
    qpt_number: int = Field(None, description="Number of q-points")
    have_dielectric: bool = Field(
        None, description="Whether the dielectric tensor is present"
    )
    have_bec: bool = Field(
        None, description="Whether the Born effective charges are present"
    )
    have_quadrupoles: bool = Field(
        None, description="Whether the quadrupoles are present"
    )
    have_efield: bool = Field(
        None, description="Whether the derivative wrt the electric field is present"
    )
    add_lr: int = Field(None, description="Treatment of long-range part of V1scf")
    qdam: float = Field(None, description="Damping factor for Gaussian Filter")

    @classmethod
    def from_abinit_logdvdb(
        cls,
        logfile: str | Path,
    ) -> CalculationOutput:
        """
        Create an Mrgdv output document from the merged Abinit out_dv file.

        Parameters
        ----------
        logfile: str or Path
            The path to the log of the mrgdv calculation.

        Returns
        -------
        The Mrgdv calculation output document.
        """
        with open(logfile) as f:
            for line in f:
                if "DVDB version" in line:
                    dvdb_version = int(line.split()[2])
                elif "Number of v1scf potentials" in line:
                    v1scf_number = int(line.split()[4])
                elif "Number of q-points in DVDB" in line:
                    qpt_number = int(line.split()[5])
                elif "Have dielectric tensor" in line:
                    have_dielectric = "yes" in line
                elif "Have Born effective charges" in line:
                    have_bec = "yes" in line
                elif "Have quadrupoles" in line:
                    have_quadrupoles = "yes" in line
                elif "Have electric field" in line:
                    have_efield = "yes" in line
                elif "Treatment of long-range part in V1scf" in line:
                    add_lr = int(line.split()[7])
                elif "Damping factor for Gaussian filter" in line:
                    qdam = float(line.split()[6])

        return cls(
            dvdb_version=dvdb_version,
            v1scf_number=v1scf_number,
            qpt_number=qpt_number,
            have_dielectric=have_dielectric,
            have_bec=have_bec,
            have_quadrupoles=have_quadrupoles,
            have_efield=have_efield,
            add_lr=add_lr,
            qdam=qdam,
        )


class Calculation(BaseModel):
    """Full Mrgdv calculation (inputs) and outputs.

    Parameters
    ----------
    dir_name: str
        The directory for this Mrgdv calculation
    has_mrgdv_completed: .TaskState
        Whether Mrgdv completed the merge successfully
    output: .CalculationOutput
        The Mrgdv calculation output
    completed_at: str
        Timestamp for when the merge was completed
    output_file_paths: Dict[str, str]
        Paths (relative to dir_name) of the Mrgdv output files
        associated with this calculation
    """

    dir_name: str = Field(None, description="The directory for this Abinit calculation")
    has_mrgdv_completed: TaskState = Field(
        None, description="Whether Abinit completed the calculation successfully"
    )
    output: Optional[CalculationOutput] = Field(
        None, description="The Abinit calculation output"
    )
    completed_at: str = Field(
        None, description="Timestamp for when the calculation was completed"
    )
    output_file_paths: Optional[dict[str, str]] = Field(
        None,
        description="Paths (relative to dir_name) of the Abinit output files "
        "associated with this calculation",
    )

    @classmethod
    def from_abinit_files(
        cls,
        dir_name: Path | str,
        task_name: str,
        abinit_outdvdb_file: Path | str = "out_dvdb",
        abinit_mrglog_file: Path | str = "run.log",
    ) -> tuple[Calculation, dict[MrgdvdbObject, dict]]:
        """
        Create a Mrgdv calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name: Path or str
            The directory containing the calculation outputs.
        task_name: str
            The task name.
        abinit_outdvdb_file: Path or str
            Path to the merged dv file, relative to dir_name.
        abinit_mrglog_file: Path or str
            Path to the main log of mrgdv job, relative to dir_name.

        Returns
        -------
        .Calculation
            A Mrgdv calculation document.
        """
        dir_name = Path(dir_name)
        abinit_outdvdb_file = dir_name / Path(abinit_outdvdb_file)
        # abinit_mrglog_file = dir_name.parent / abinit_mrglog_file

        output_doc = None
        mrgdv_objects: dict[MrgdvdbObject, Any] = {}
        if abinit_outdvdb_file.exists() and Path(abinit_mrglog_file).exists():
            mrgdv_objects[MrgdvdbObject.DVDBFILE] = AbinitStoredFile.from_file(  # type: ignore[index]
                filepath=abinit_outdvdb_file, data_type=bytes
            )
            output_doc = CalculationOutput.from_abinit_logdvdb(abinit_mrglog_file)

            completed_at = str(
                datetime.fromtimestamp(
                    os.stat(abinit_outdvdb_file).st_mtime, tz=timezone.utc
                )
            )

        report = None
        has_mrgdv_completed = TaskState.FAILED
        try:
            report = get_mrgdv_report(logfile=abinit_mrglog_file)

            if report["run_completed"] and abinit_outdvdb_file.exists():
                has_mrgdv_completed = TaskState.SUCCESS

        # except (dvError, Exception) as exc:
        except Exception as exc:
            msg = f"{cls} exception while parsing mrgdv event_report:\n{exc}"
            logger.critical(msg)
            logging.exception(msg)

        return (
            cls(
                dir_name=str(dir_name),
                task_name=task_name,
                has_mrgdv_completed=has_mrgdv_completed,
                completed_at=completed_at,
                output=output_doc,
            ),
            mrgdv_objects,
        )


class OutputDoc(BaseModel):
    """Summary of the outputs for a Mrgdv calculation.

    Parameters
    ----------
    dvdb_version: int
        The DVDB file version
    v1scf_number: int
        Number of V1SCF potential
    qpt_number: int
        Number of q-points
    have_dielectric: bool
        Whether the dielectric tensor is present
    have_bec: bool
        Whether the Born effective charges are present
    have_quadrupoles: bool
        Whether the quadrupoles are present
    have_efield: bool
        Whether the derivative wrt the electric field is present
    add_lr: int
        Treatment of long-range part of V1scf
    qdam: float
        Damping factor for Gaussian Filter
    """

    dvdb_version: Optional[int] = Field(None, description="The DVDB file version")
    v1scf_number: Optional[int] = Field(None, description="Number of V1SCF potential")
    qpt_number: Optional[int] = Field(None, description="Number of q-points")
    have_dielectric: Optional[bool] = Field(
        None, description="Whether the dielectric tensor is present"
    )
    have_bec: Optional[bool] = Field(
        None, description="Whether the Born effective charges are present"
    )
    have_quadrupoles: Optional[bool] = Field(
        None, description="Whether the quadrupoles are present"
    )
    have_efield: Optional[bool] = Field(
        None, description="Whether the derivative wrt the electric field is present"
    )
    add_lr: Optional[int] = Field(
        None, description="Treatment of long-range part of V1scf"
    )
    qdam: Optional[float] = Field(
        None, description="Damping factor for Gaussian Filter"
    )

    @classmethod
    def from_abinit_calc_doc(cls, calc_doc: Calculation) -> OutputDoc:
        """Create a summary from an abinit CalculationDocument.

        Parameters
        ----------
        calc_doc: .Calculation
            A Mrgdv calculation document.

        Returns
        -------
        .OutputDoc
            The calculation output summary.
        """
        return cls(
            dvdb_version=calc_doc.output.dvdb_version,
            v1scf_number=calc_doc.output.v1scf_number,
            qpt_number=calc_doc.output.qpt_number,
            have_dielectric=calc_doc.output.have_dielectric,
            have_bec=calc_doc.output.have_bec,
            have_quadrupoles=calc_doc.output.have_quadrupoles,
            have_efield=calc_doc.output.have_efield,
            add_lr=calc_doc.output.add_lr,
            qdam=calc_doc.output.qdam,
        )


class MrgdvdbTaskDoc(StructureMetadata):
    """Definition of task document about an Mrgdv Job.

    Parameters
    ----------
    dir_name: str
        The directory for this Abinit task
    completed_at: str
        Timestamp for when this task was completed
    output: .OutputDoc
        The output of the final calculation
    state: .TaskState
        State of this task
    included_objects: List[.MrgdvdbObject]
        List of Abinit objects included with this task document
    abinit_objects: Dict[.MrgdvdbObject, Any]
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
    output: Optional[OutputDoc] = Field(
        None, description="The output of the final calculation"
    )
    state: Optional[TaskState] = Field(None, description="State of this task")
    included_objects: Optional[list[MrgdvdbObject]] = Field(
        None, description="List of Mrgdv objects included with this task document"
    )
    mrgdv_objects: Optional[dict[MrgdvdbObject, Any]] = Field(
        None, description="Mrgdv objects associated with this task"
    )
    task_label: Optional[str] = Field(None, description="A description of the task")
    dvdb_version: Optional[int] = Field(None, description="The DVDB file version")
    v1scf_number: Optional[int] = Field(None, description="Number of V1SCF potential")
    qpt_number: Optional[int] = Field(None, description="Number of q-points")
    have_dielectric: Optional[bool] = Field(
        None, description="Whether the dielectric tensor is present"
    )
    have_bec: Optional[bool] = Field(
        None, description="Whether the Born effective charges are present"
    )
    have_quadrupoles: Optional[bool] = Field(
        None, description="Whether the quadrupoles are present"
    )
    have_efield: Optional[bool] = Field(
        None, description="Whether the derivative wrt the electric field is present"
    )
    add_lr: Optional[int] = Field(
        None, description="Treatment of long-range part of V1scf"
    )
    qdam: Optional[float] = Field(
        None, description="Damping factor for Gaussian Filter"
    )
    tags: Optional[list[str]] = Field(
        None, description="Metadata tags for this task document"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        additional_fields: dict[str, Any] = None,
        **abinit_calculation_kwargs,
    ) -> MrgdvdbTaskDoc:
        """Create a task document from a directory containing Abinit/Mrgdv files.

        Parameters
        ----------
        dir_name: Path or str
            The path to the folder containing the calculation outputs.
        additional_fields: Dict[str, Any]
            Dictionary of additional fields to add to output document.
        **abinit_calculation_kwargs
            Additional parsing options that will be passed to the
            :obj:`.Calculation.from_abinit_files` function.

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

        calcs_reversed = []
        all_mrgdv_objects = []
        for task_name, files in task_files.items():
            calc_doc, mrgdv_objects = Calculation.from_abinit_files(
                dir_name, task_name, **files, **abinit_calculation_kwargs
            )
            calcs_reversed.append(calc_doc)
            all_mrgdv_objects.append(mrgdv_objects)

        tags = additional_fields.get("tags")

        dir_name = get_uri(dir_name)  # convert to full uri path

        # only store objects from last calculation
        # TODO: make this an option
        mrgdv_objects = all_mrgdv_objects[-1]
        included_objects = None
        if mrgdv_objects:
            included_objects = list(mrgdv_objects)

        # rewrite the original structure save!

        data = {
            "calcs_reversed": calcs_reversed,
            "completed_at": calcs_reversed[-1].completed_at,
            "dir_name": dir_name,
            "included_objects": included_objects,
            "mrgdv_objects": mrgdv_objects,
            # "meta_structure": calcs_reversed[-1].output.structure,
            "state": calcs_reversed[-1].has_mrgdv_completed,
            # "structure": calcs_reversed[-1].output.structure,
            "dvdb_version": calcs_reversed[-1].output.dvdb_version,
            "v1scf_number": calcs_reversed[-1].output.v1scf_number,
            "qpt_number": calcs_reversed[-1].output.qpt_number,
            "have_dielectric": calcs_reversed[-1].output.have_dielectric,
            "have_bec": calcs_reversed[-1].output.have_bec,
            "have_quadrupoles": calcs_reversed[-1].output.have_quadrupoles,
            "have_efield": calcs_reversed[-1].output.have_efield,
            "add_lr": calcs_reversed[-1].output.add_lr,
            "qdam": calcs_reversed[-1].output.qdam,
            "tags": tags,
        }

        doc = cls(**data)
        # doc = doc.model_copy(update=data)
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
