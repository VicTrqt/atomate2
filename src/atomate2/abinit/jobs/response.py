"""Jobs for running ABINIT response to perturbations."""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from abipy.flowtk.events import (
    AbinitCriticalWarning,
    NscfConvergenceWarning,
    ScfConvergenceWarning,
)
from jobflow import Flow, Job, Response, job

from atomate2.abinit.jobs.base import BaseAbinitMaker, abinit_job
from atomate2.abinit.powerups import update_user_abinit_settings
from atomate2.abinit.sets.response import (
    DdeSetGenerator,
    DdkSetGenerator,
    DteSetGenerator,
    PhononSetGenerator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from abipy.abio.inputs import AbinitInput
    from pymatgen.core.structure import Structure
    from pymatgen.io.abinit.abiobjects import KSampling

    from atomate2.abinit.sets.base import AbinitInputGenerator, get_ksampling
    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "DdeMaker",
    "DdkMaker",
    "DteMaker",
    "PhononResponseMaker",
    "ResponseMaker",
    "generate_dde_perts",
    "generate_dte_perts",
    "generate_phonon_perts",
    "run_rf",
]


@dataclass
class ResponseMaker(BaseAbinitMaker):
    """Maker for a Response Function ABINIT calculation job.

    Parameters
    ----------
    calc_type : str
        The type of RF.
    name : str
        The job name.
    """

    calc_type: str = "RF"
    name: str = "RF calculation"
    task_document_kwargs: dict = field(
        default_factory=lambda: {"files_to_store": ["DDB"]}
    )
    input_set_generator: AbinitInputGenerator
    stop_jobflow_on_failure: bool = True

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )

    @abinit_job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        perturbation: dict | None = None,
    ) -> Job:
        """
        Run a RF ABINIT job. The type of RF is defined by self.calc_type.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object
        perturbation : dict
            Direction of the perturbation for the RF calculation.
            Abipy format.
        """
        if perturbation:
            self.input_set_generator.factory_kwargs.update(
                {f"{self.calc_type.lower()}_pert": perturbation}
            )

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
        )


@dataclass
class DdkMaker(ResponseMaker):
    """Maker to create a job with a DDK ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "DDK"
    name: str = "DDK calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DdkSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
        ScfConvergenceWarning,
    )


@dataclass
class DdeMaker(ResponseMaker):
    """Maker to create a job with a DDE ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "DDE"
    name: str = "DDE calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DdeSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@dataclass
class DteMaker(ResponseMaker):
    """Maker to create a job with a DTE ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "DTE"
    name: str = "DTE calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DteSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@dataclass
class PhononResponseMaker(ResponseMaker):
    """Maker to create a job with a Phonon ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "Phonon"
    name: str = "Phonon calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=PhononSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@job
def generate_dde_perts(
    gsinput: AbinitInput,
    # TODO: or gsinput via prev_outputs?
    use_symmetries: bool | None = False,
) -> dict[str, dict]:
    """
    Generate the perturbations for the DDE calculations.

    Parameters
    ----------
    gsinput : an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    use_symmetries : True if only the irreducible perturbations should
        be returned, False otherwise.
    """
    if use_symmetries:
        gsinput = gsinput.deepcopy()
        gsinput.pop_vars(["autoparal"])
        gsinput.pop_par_vars(all=True)
        perts = gsinput.abiget_irred_ddeperts()  # TODO: quid manager?
    else:
        perts = [{"idir": 1}, {"idir": 2}, {"idir": 3}]

    outputs = {"perts": perts}
    # to make the dir of generate... accessible
    outputs["dir_name"] = Path(os.getcwd())
    return outputs


@job
def generate_dte_perts(
    gsinput: AbinitInput,
    # TODO: or gsinput via prev_outputs?
    skip_permutations: bool | None = False,
    phonon_pert: bool | None = False,
) -> dict[str, dict]:
    """
    Generate the perturbations for the DTE calculations.

    Parameters
    ----------
    gsinput : an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    skip_permutations: Since the current version of abinit always performs
        all the permutations of the perturbations, even if only one is asked,
        if True avoids the creation of inputs that will produce duplicated outputs.
    phonon_pert: is True also the phonon perturbations will be considered.
        Default False.
    """
    # Call Abinit to get the list of irreducible perturbations
    gsinput = gsinput.deepcopy()
    gsinput.pop_vars(["autoparal"])
    gsinput.pop_par_vars(all=True)
    perts = gsinput.abiget_irred_dteperts(
        phonon_pert=phonon_pert,
    )  # TODO: quid manager?

    if skip_permutations:
        perts_to_skip: list = []
        reduced_perts = []
        for pert in perts:
            p = (
                (pert.i1pert, pert.i1dir),
                (pert.i2pert, pert.i2dir),
                (pert.i3pert, pert.i3dir),
            )
            if p not in perts_to_skip:
                reduced_perts.append(pert)
                perts_to_skip.extend(itertools.permutations(p))

        perts = reduced_perts

    outputs = {"perts": perts}
    outputs["dir_name"] = Path(os.getcwd())  # to make the dir of run_rf accessible
    return outputs


@job
def generate_phonon_perts(
    gsinput: AbinitInput,
    ngqpt: list | tuple | None = None,
    qptopt: int | None = 1,
    qpt_list: list[list] | None = None,
    with_wfq: bool = False,
) -> dict[str, list[Any] | tuple[Any, ...] | Any]:
    """
    Generate the qpt-list and perturbations for the phonon calculations.

    Parameters
    ----------
    gsinput : an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    ngqpt : list or tuple
        Monkhorst-Pack divisions for the phonon q-mesh.
        Default is the same as the one used in the GS calculation.
        Must be a sub-mesh of the k-mesh used for electrons.
    qptopt : int
        Option for the q-point generation.
    qpt_list: list
        q-point for the phonon calculations.
    with_wfq: bool
        True if a wfq_maker is provided for k+q computations.
        Not yet implemented, so default is False.

    Returns
    -------
    dict
        A dictionary with the perturbations, the ngqpt and the
        output directory name.
    """
    gsinput = gsinput.deepcopy()
    gsinput.pop_vars(["autoparal"])
    outputs = {}
    if qpt_list is None:
        qpt_list = gsinput.abiget_ibz(
            ngkpt=ngqpt, shiftk=[0, 0, 0], kptopt=qptopt
        ).points
        outputs["ngqpt"] = ngqpt if ngqpt else gsinput["ngkpt"]
    else:
        outputs["ngqpt"] = [1, 1, 1]
    qpt_list = [qpt_list] if isinstance(qpt_list[0], int | float) else qpt_list
    perturbations = list()
    outdirs = list()
    for q in qpt_list:
        perts = gsinput.abiget_irred_phperts(qpt=q)
        perturbations.append(perts)
        outdirs.append(Path(os.getcwd()))  # to make the dir accessible
        # when a wfq_maker will be available something like ... can be added here
        # if q not in kpt_list and with_wfq:
        #     wfq_job = wfq_maker.make(q=q, prev_outputs=prev_outputs)
        #     outputs["dirs"].append(wfq_job.output.dir_name)
        # and the last if removed
    outputs["perts"] = list(np.hstack(perturbations))
    outputs["dir_name"] = list(np.hstack(outdirs))
    if any(np.array(gsinput["ngkpt"]) % np.array(outputs["ngqpt"])) and not with_wfq:
        raise ValueError("q-points are not commensurate with k-points.")
    return outputs


@job
def generate_perts(
    gsinput: AbinitInput,
    scf_output,
    skip_dte_permutations: bool | None = False,
    use_dde_symmetries: bool | None = False,
    ngqpt: list | tuple | None = None,
    qptopt: int | None = 1,
    qpt_list: list[list] | None = None,
    user_qpoints_settings: dict | KSampling | None = None,
    ddk_maker: ResponseMaker | None = None,
    dde_maker: ResponseMaker | None = None,
    phonon_maker: ResponseMaker | None = None,
    dte_maker: ResponseMaker | None = None,
) -> Flow:
    """
    Generate the perturbations for the DTE calculations.

    Parameters
    ----------
    gsinput : an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    skip_dte_permutations: Since the current version of abinit always performs
        all the permutations of the perturbations, even if only one is asked,
        if True avoids the creation of inputs that will produce duplicated outputs.
    """

    if all(not m for m in [ddk_maker, dde_maker, phonon_maker, dte_maker]):
        raise ValueError("At least one of the response makers should be defined")

    cwd = Path.cwd()
    outputs = {"perts": {}, "dirs": {}}

    ddk_jobs = []
    dde_jobs = []
    ph_jobs = []
    dte_jobs = []

    if not isinstance(scf_output, (list, tuple)):
        scf_output = [scf_output]

    gsinput = gsinput.deepcopy()
    gsinput.pop_vars(["autoparal"])
    gsinput.pop_par_vars(all=True)

    # DDK
    if ddk_maker:
        # the use of symmetries is not implemented for DDK
        perturbations = [{"idir": 1}, {"idir": 2}, {"idir": 3}]
        ddk_jobs = []
        for ipert, pert in enumerate(perturbations):
            ddk_job = ddk_maker.make(
                perturbation=pert,
                prev_outputs=scf_output,
            )
            ddk_job.append_name(f"{ipert + 1}/{len(perturbations)}")

            ddk_jobs.append(ddk_job)

        outputs["perts"]["ddk"] = [j.output for j in ddk_jobs]
        outputs["dirs"]["ddk"] = [j.output.dir_name for j in ddk_jobs]

    # DDE
    if dde_maker:
        if not ddk_maker:
            raise ValueError("DDK maker is required to run DDE")
        if use_dde_symmetries:
            gsinput = gsinput.deepcopy()
            gsinput.pop_vars(["autoparal"])
            gsinput.pop_par_vars(all=True)
            dde_perts = gsinput.abiget_irred_ddeperts(workdir=cwd / "dde")  # TODO: quid manager?
        else:
            dde_perts = [{"idir": 1}, {"idir": 2}, {"idir": 3}]

        dde_prev = scf_output + [j.output.dir_name for j in ddk_jobs]
        dde_jobs = get_jobs(dde_perts, rf_maker=dde_maker, prev_outputs=dde_prev)

        outputs["perts"]["dde"] = [j.output for j in dde_jobs]
        outputs["dirs"]["dde"] = [j.output.dir_name for j in dde_jobs]

    # Phonons
    if phonon_maker:
        if qpt_list is not None and ngqpt is not None:
            raise ValueError("qpt_list and ngqpt can't be used together")

        if qpt_list is None:
            if ngqpt is not None:
                ngqpt = np.array(ngqpt)
            elif user_qpoints_settings is not None:
                ksampling = get_ksampling(structure=gsinput.structure, user_kpoints_settings=user_qpoints_settings, force_gamma=True)
                if "ngkpt" not in ksampling.abivars:
                    raise RuntimeError(f"Could not determine ngqpt from ksampling {ksampling}")
                ngqpt = ksampling.abivars["ngkpt"]
            else:
                ngqpt = np.array(gsinput["ngkpt"])

            qpt_list = gsinput.abiget_ibz(ngkpt=ngqpt, shiftk=(0, 0, 0), kptopt=qptopt).points

        # check that qpt are consistent with kpt grid
        if ngqpt is None or any(gsinput["ngkpt"] % ngqpt != 0):
            # find which q points are needed and build nscf inputs to calculate the WFQ
            kpts = gsinput.abiget_ibz(shiftk=(0, 0, 0), kptopt=3).points.tolist()
            nscf_qpt = []
            for q in qpt_list:
                if list(q) not in kpts:
                    raise ValueError("At least one selected qpoint is not commensurate with the kpt grid")
                    # nscf_qpt.append(q)

        ph_perts = list()
        for qpt_i, q in enumerate(qpt_list):
            perts = gsinput.abiget_irred_phperts(qpt=q, workdir=cwd / f"q_{qpt_i}")
            ph_perts.extend(perts)
            # when a wfq_maker will be available something like ... can be added here
            # if q not in kpt_list and with_wfq:
            #     wfq_job = wfq_maker.make(q=q, prev_outputs=prev_outputs)
            #     outputs["dirs"].append(wfq_job.output.dir_name)
            # and the last if removed

        ph_jobs = get_jobs(ph_perts, rf_maker=phonon_maker, prev_outputs=scf_output, is_phonon=True)

        outputs["perts"]["phonon"] = [j.output for j in ph_jobs]
        outputs["dirs"]["phonon"] = [j.output.dir_name for j in ph_jobs]

    # DTE
    if dte_maker:
        dte_perts = gsinput.abiget_irred_dteperts(
            phonon_pert=phonon_maker is not None,
            workdir=cwd / f"dte"
        )  # TODO: quid manager?

        if skip_dte_permutations:
            perts_to_skip: list = []
            reduced_perts = []
            for pert in dte_perts:
                p = (
                    (pert.i1pert, pert.i1dir),
                    (pert.i2pert, pert.i2dir),
                    (pert.i3pert, pert.i3dir),
                )
                if p not in perts_to_skip:
                    reduced_perts.append(pert)
                    perts_to_skip.extend(itertools.permutations(p))

            dte_perts = reduced_perts

        dde_jobs = [update_user_abinit_settings(ddej, {"prtwf": 1}) for ddej in dde_jobs]
        ph_jobs = [update_user_abinit_settings(pj, {"prtwf": 1}) for pj in ph_jobs]

        dte_prev = scf_output + [j.output.dir_name for j in dde_jobs] + [j.output.dir_name for j in ph_jobs]
        dte_jobs = get_jobs(dte_perts, rf_maker=dte_maker, prev_outputs=dte_prev)
        outputs["perts"]["dte"] = [j.output for j in dte_jobs]
        outputs["dirs"]["dte"] = [j.output.dir_name for j in dte_jobs]

    jobs = ddk_jobs + dde_jobs + ph_jobs + dte_jobs

    rf_flow = Flow(jobs, outputs)
    return Response(replace=rf_flow, output={"dir_name": cwd})  # TODO what is the output here?


def get_jobs(
    perturbations: list[dict],
    rf_maker: ResponseMaker,
    prev_outputs: str | list[str] | None = None,
    is_phonon: bool = False
) -> list[Job]:
    """
    Run the RF calculations.

    Parameters
    ----------
    perturbations : a list of dict with the direction of the perturbation
        under the Abipy format.
    rf_maker : Maker to create a job with a Response Function ABINIT calculation.
    prev_outputs : a list of previous output directories
    """
    rf_jobs = []

    for ipert, pert in enumerate(perturbations):
        rf_job = rf_maker.make(
            perturbation=pert,
            prev_outputs=prev_outputs,
        )

        if is_phonon:
            qpt_str = f"{pert['qpt'][0]:.2f},{pert['qpt'][1]:.2f},{pert['qpt'][2]:.2f}"
            rf_job.append_name(f", q = {qpt_str} ({ipert+1}/{len(perturbations)})")
        else:
            rf_job.append_name(f"{ipert+1}/{len(perturbations)}")

        rf_jobs.append(rf_job)

    return rf_jobs


@job
def run_rf(
    perturbations: list[dict],
    rf_maker: ResponseMaker,
    prev_outputs: str | list[str] | None = None,
) -> Flow:
    """
    Run the RF calculations.

    Parameters
    ----------
    perturbations : a list of dict with the direction of the perturbation
        under the Abipy format.
    rf_maker : Maker to create a job with a Response Function ABINIT calculation.
    prev_outputs : a list of previous output directories
    """
    rf_jobs = []
    is_phonon = isinstance(rf_maker, PhononResponseMaker)
    outputs: dict[str, Any] = {"dirs": []}

    if isinstance(rf_maker, DdeMaker | DteMaker | PhononResponseMaker):
        # Flatten the list of previous outputs dir
        # prev_outputs = [item for sublist in prev_outputs for item in sublist]
        prev_outputs = list(np.hstack(prev_outputs))

    for ipert, pert in enumerate(perturbations):
        rf_job = rf_maker.make(
            perturbation=pert,
            prev_outputs=prev_outputs,
        )

        if is_phonon:
            qpt_str = f"{pert['qpt'][0]:.2f},{pert['qpt'][1]:.2f},{pert['qpt'][2]:.2f}"
            rf_job.append_name(f", q = {qpt_str} ({ipert+1}/{len(perturbations)})")
        else:
            rf_job.append_name(f"{ipert+1}/{len(perturbations)}")

        rf_jobs.append(rf_job)
        outputs["dirs"].append(rf_job.output.dir_name)  # TODO: determine outputs

    outputs["dir_name"] = Path(os.getcwd())  # to make the dir of run_rf accessible
    rf_flow = Flow(rf_jobs, outputs)

    return Response(replace=rf_flow)
