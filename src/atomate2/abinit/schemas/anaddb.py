"""Core definitions of Abinit calculations documents."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

# from typing import Type, TypeVar, Union, Optional, List
from typing import Any, Optional, Union

import abipy.core.abinit_units as abu
import numpy as np
from abipy.dfpt.anaddbnc import AnaddbNcFile
from abipy.dfpt.phonons import PhononBands, PhononDos
from abipy.flowtk import events
from abipy.flowtk.utils import File
from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos as pmgPhononDos

from atomate2.abinit.schemas.calculation import AbinitObject
from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.utils.common import get_event_report
from atomate2.common.schemas.phonons import ThermalDisplacementData
from atomate2.utils.path import get_uri

logger = logging.getLogger(__name__)


class OutputDoc(BaseModel):
    """Summary of the outputs for a anaddb calculation.

    Parameters
    ----------
    structure: Structure
        The final pymatgen Structure of the final system
    dijk: list (3x3x3)
        The conventional static SHG tensor in pm/V (Chi^(2)/2)
    epsinf: list (3x3)
        The electronic contribution to the dielectric tensor
    phonon_bandstructure: PhononBandStructureSymmLine
        The phonon band structure object.
    phonon_dos: PhononDos
        The phonon density of states object.
    free_energies: list
        The vibrational part of the free energies in J/mol per
        formula unit for temperatures in temperature_list
    heat_capacities: list
        The heat capacities in J/K/mol per
        formula unit for temperatures in temperature_list
    internal_energies: list
        The internal energies in J/mol per
        formula unit for temperatures in temperature_list
    entropies: list
        The entropies in J/(K*mol) per formula unit
        for temperatures in temperature_list
    temperatures: list
        The temperatures at which the vibrational
        part of the free energies and other properties have been computed
    volume_per_formula_unit: float
        The volume per formula unit in Angstrom**3
    formula_units: int
        The number of formula units per cell
    has_imaginary_modes: bool
        Whether the structure has imaginary modes
    born: list
        The Born charges as computed from phonopy. Only for symmetrically
        different atoms
    epsilon_static: Matrix3D
        The high-frequency dielectric constant
    """

    structure: Union[Structure] = Field(
        None, description="The final structure from the calculation"
    )
    dijk: Optional[list] = Field(
        None, description="Conventional SHG tensor in pm/V (Chi^(2)/2)"
    )
    epsinf: Optional[list] = Field(
        None, description="Electronic contribution to the dielectric tensor"
    )
    phonon_bandstructure: Optional[PhononBandStructureSymmLine] = Field(
        None,
        description="Phonon band structure object.",
    )

    phonon_dos: Optional[pmgPhononDos] = Field(
        None,
        description="Phonon density of states object.",
    )

    free_energies: Optional[list[float]] = Field(
        None,
        description="vibrational part of the free energies in J/mol per "
        "formula unit for temperatures in temperature_list",
    )

    heat_capacities: Optional[list[float]] = Field(
        None,
        description="heat capacities in J/K/mol per "
        "formula unit for temperatures in temperature_list",
    )

    internal_energies: Optional[list[float]] = Field(
        None,
        description="internal energies in J/mol per "
        "formula unit for temperatures in temperature_list",
    )
    entropies: Optional[list[float]] = Field(
        None,
        description="entropies in J/(K*mol) per formula unit"
        "for temperatures in temperature_list ",
    )

    temperatures: Optional[list[int]] = Field(
        None,
        description="temperatures at which the vibrational"
        " part of the free energies"
        " and other properties have been computed",
    )

    volume_per_formula_unit: Optional[float] = Field(
        None, description="volume per formula unit in Angstrom**3"
    )

    formula_units: Optional[int] = Field(None, description="Formula units per cell")

    has_imaginary_modes: Optional[bool] = Field(
        None, description="if true, structure has imaginary modes"
    )

    born: Optional[list[Matrix3D]] = Field(
        None,
        description="Born charges as computed from phonopy. Only for symmetrically "
        "different atoms",
    )

    epsilon_static: Optional[Matrix3D] = Field(
        None, description="The high-frequency dielectric constant"
    )

    thermal_displacement_data: Optional[ThermalDisplacementData] = Field(
        None,
        description="Includes all data of the computation of the thermal displacements",
    )

    @classmethod
    def from_abinit_files(
        cls,
        dir_name: Path | str,
        abinit_anaddb_file: Path | str = "out_anaddb.nc",
        abinit_analog_file: Path | str = "run.log",  # noqa: ARG003
        abinit_phbst_file: Path | str = "out_PHBST.nc",
        abinit_phdos_file: Path | str = "out_PHDOS.nc",
    ) -> OutputDoc:
        """
        Create a anaddb calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name: Path or str
            The directory containing the calculation outputs.
        abinit_anaddb_file: Path or str
            Path to the merged DDB file, relative to dir_name.
        abinit_analog_file: Path or str
            Path to the main log of anaddb job, relative to dir_name.
        abinit_phbst_file: Path or str
            Path to the PHBST file, relative to dir_name
        abinit_phdos_file: Path or str
            Path to the PHDOS file, relative to dir_name
        other_files: dict


        Returns
        -------
        .OuputDoc
            An anaddb output document.
        """
        dir_name = Path(dir_name)
        abinit_anaddb_file = dir_name / abinit_anaddb_file
        abinit_phbst_file = dir_name / abinit_phbst_file
        abinit_phdos_file = dir_name / abinit_phdos_file

        if abinit_anaddb_file.exists():
            abinit_anaddb = AnaddbNcFile.from_file(abinit_anaddb_file)
        else:
            raise RuntimeError(
                f"The file {abinit_anaddb_file} is missing and is required \
                to generate the output document"
            )
        if abinit_phbst_file.exists():
            abinit_phbst = PhononBands.from_file(abinit_phbst_file)
        else:
            abinit_phbst = None
        if abinit_phdos_file.exists():
            abinit_phdos = PhononDos.as_phdos(str(abinit_phdos_file))
        else:
            abinit_phdos = None

        structure = abinit_anaddb.structure

        if abinit_phbst:
            phonon_bandstructure = abinit_phbst.to_pymatgen()
            phonon_bandstructure.labels_dict = {
                k.strip("$"): v for k, v in phonon_bandstructure.labels_dict.items()
            }
        else:
            phonon_bandstructure = None
        phonon_dos = abinit_phdos.to_pymatgen() if abinit_phbst else None

        if phonon_dos:
            temperatures = [int(t) for t in abinit_phdos.get_free_energy().mesh]
            free_energies = [
                phonon_dos.helmholtz_free_energy(temp, structure=structure)
                for temp in temperatures
            ]
            heat_capacities = [
                phonon_dos.cv(temp=temp, structure=structure) for temp in temperatures
            ]
            internal_energies = [
                phonon_dos.internal_energy(temp, structure=structure)
                for temp in temperatures
            ]
            entropies = [
                phonon_dos.entropy(temp, structure=structure) for temp in temperatures
            ]
        else:
            temperatures = None
            free_energies = None
            heat_capacities = None
            internal_energies = None
            entropies = None
        born = getattr(abinit_anaddb, "bec", None)
        born = born.values if born else None

        formula_units = (
            structure.composition.num_atoms
            / structure.composition.reduced_composition.num_atoms
        )
        volume_per_formula_unit = structure.volume / formula_units

        has_imaginary_modes = (
            phonon_bandstructure.has_imaginary_freq() if phonon_bandstructure else None
        )

        # for pm/V units (SI)
        dijk = (
            list(
                abinit_anaddb.dchide
                * 16
                * np.pi**2
                * abu.Bohr_Ang**2
                * 1e-8
                * abu.eps0
                / abu.e_Cb
            )
            if abinit_anaddb.dchide is not None and abinit_anaddb.dchide.any()
            else None
        )
        epsinf = (
            list(abinit_anaddb.epsinf)
            if abinit_anaddb.epsinf is not None and abinit_anaddb.epsinf.any()
            else None
        )
        return cls(
            structure=structure,
            phonon_bandstructure=phonon_bandstructure,
            phonon_dos=phonon_dos,
            free_energies=free_energies,
            heat_capacities=heat_capacities,
            internal_energies=internal_energies,
            entropies=entropies,
            temperatures=temperatures,
            volume_per_formula_unit=volume_per_formula_unit,
            formula_units=formula_units,
            has_imaginary_modes=has_imaginary_modes,
            born=born,
            dijk=dijk,
            epsinf=epsinf,
        )


class AnaddbTaskDoc(StructureMetadata, extra="allow"):  # type: ignore[call-arg]
    """Definition of task document about an anaddb Job.

    Parameters
    ----------
    dir_name: str
        The directory for this Abinit task
    completed_at: str
        Timestamp for when this task was completed
    output: .OutputDoc
        The output of the final calculation
    structure: Structure
        Final output structure from the task
    included_objects: List[.AbinitObject]
        List of Abinit objects included with this task document
    abinit_objects: Dict[.AbinitObject, Any]
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
    structure: Union[Structure] = Field(
        None, description="Final output atoms from the task"
    )
    event_report: Optional[events.EventReport] = Field(
        None, description="Event report of this abinit job."
    )
    included_objects: Optional[list[AbinitObject]] = Field(
        None, description="List of Abinit objects included with this task document"
    )
    abinit_objects: Optional[dict[AbinitObject, Any]] = Field(
        None, description="Abinit objects associated with this task"
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
        abinit_phbst_file: Path | str = "out_PHBST.nc",
        abinit_phdos_file: Path | str = "out_PHDOS.nc",
        abinit_analog_file: Path | str = "run.log",
        files_to_store: list | None = None,
        **output_doc_kwargs,
    ) -> AnaddbTaskDoc:
        """Create a task document from a directory containing Abinit/anaddb files.

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
        .AnaddbTaskDoc
            A task document for the calculation.
        """
        logger.info(f"Getting task doc in: {dir_name}")

        if additional_fields is None:
            additional_fields = {}

        dir_name = Path(dir_name)
        task_files = _find_abinit_files(dir_name)

        if len(task_files) == 0:
            raise FileNotFoundError("No anaddb files found!")
        if len(task_files) > 1:
            raise RuntimeError(
                f"Only one anaddb calculation expected. Found {len(task_files)}"
            )

        std_task_files = next(iter(task_files.values()))

        report = get_event_report(ofile=File(abinit_analog_file))
        if not report.run_completed:
            raise RuntimeError("Anaddb did not complete successfully")

        output_doc = OutputDoc.from_abinit_files(
            dir_name, **std_task_files, **output_doc_kwargs
        )

        phbst_filepath = dir_name / abinit_phbst_file
        abinit_objects: dict[AbinitObject, Any] = {}
        if phbst_filepath.exists() and "PHBST" in files_to_store:
            abinit_objects[AbinitObject.PHBSTFILE] = AbinitStoredFile.from_file(  # type: ignore[index]
                filepath=phbst_filepath, data_type=bytes
            )
        phdos_filepath = dir_name / abinit_phdos_file
        if phdos_filepath.exists() and "PHDOS" in files_to_store:
            abinit_objects[AbinitObject.PHDOSFILE] = AbinitStoredFile.from_file(  # type: ignore[index]
                filepath=phdos_filepath, data_type=bytes
            )
        completed_at = str(
            datetime.fromtimestamp(
                os.stat(abinit_analog_file).st_mtime, tz=timezone.utc
            )
        )

        tags = additional_fields.get("tags")

        dir_name = get_uri(dir_name)  # convert to full uri path

        included_objects = None
        if abinit_objects:
            included_objects = list(abinit_objects.keys())

        data = {
            "dir_name": dir_name,
            "completed_at": completed_at,
            "output": output_doc,
            "event_report": report,
            "included_objects": included_objects,
            "abinit_objects": abinit_objects,
            "tags": tags,
        }

        return cls.from_structure(
            structure=output_doc.structure,
            meta_structure=output_doc.structure,
            **data,
            **additional_fields,
        )


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
            if file.match(f"*outdata/out_anaddb.nc{suffix}*"):
                abinit_files["abinit_anaddb_file"] = Path(file).relative_to(path)
            elif file.match(f"*run.log{suffix}*"):
                abinit_files["abinit_analog_file"] = Path(file).relative_to(path)
            if file.match(f"*outdata/out_PHBST.nc{suffix}*"):
                abinit_files["abinit_phbst_file"] = Path(file).relative_to(path)
            if file.match(f"*outdata/out_PHDOS.nc{suffix}*"):
                abinit_files["abinit_phdos_file"] = Path(file).relative_to(path)

        return abinit_files

    # get any matching file from the root folder
    standard_files = _get_task_files(
        list(path.glob("*")) + list(path.glob("outdata/*"))
    )
    if len(standard_files) > 0:
        task_files["standard"] = standard_files

    return task_files
