"""DFPT abinit flow makers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import abipy.core.abinit_units as abu
from abipy.abio.factories import scf_for_phonons
from jobflow import Flow, Maker

from atomate2.abinit.jobs.anaddb import (
    AnaddbDfptDteMaker,
    AnaddbMaker,
    AnaddbPhBandsDOSMaker,
)
from atomate2.abinit.jobs.core import StaticMaker, StaticMakerforPhonons
from atomate2.abinit.jobs.mrgddb import MrgddbMaker
from atomate2.abinit.jobs.mrgdv import MrgdvMaker
from atomate2.abinit.jobs.response import (
    DdeMaker,
    DdkMaker,
    DteMaker,
    PhononResponseMaker,
    generate_dde_perts,
    generate_dte_perts,
    generate_phonon_perts,
    run_rf,
)
from atomate2.abinit.powerups import update_user_abinit_settings
from atomate2.abinit.sets.core import ShgStaticSetGenerator, StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.abinit.jobs.base import BaseAbinitMaker


@dataclass
class DfptFlowMaker(Maker):
    """
    Maker to generate a DFPT flow with abinit.

    The classmethods allow to tailor the flow for specific properties
        accessible via DFPT.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseAbinitMaker
        The maker to use for the static calculation.
    ddk_maker : .BaseAbinitMaker
        The maker to use for the DDK calculations.
    dde_maker : .BaseAbinitMaker
        The maker to use for the DDE calculations.
    dte_maker : .BaseAbinitMaker
        The maker to use for the DTE calculations.
    wfq_maker : .BaseAbinitMaker
        The maeker to use to compute the k+q WFs.
        Not implemented yet.
    phonon_maker : .BaseAbinitMaker
        The maker to use for the phonon calculations.
    mrgddb_maker : .Maker
        The maker to merge the DDBs.
    mrgdv_maker : .Maker
        The maker to merge the POT files.
    anaddb_maker : .Maker
        The maker to analyze the DDBs.
    use_dde_sym : bool
        True if only the irreducible DDE perturbations should be considered,
            False otherwise.
    dte_skip_permutations: Since the current version of abinit always performs
        all the permutations of the perturbations, even if only one is asked,
        if True avoids the creation of inputs that will produce duplicated outputs.
    qpt_list: list or tuple or None
        A list of q points to compute the phonon band structure.
        All the q points must be part of the k-mesh used for electrons.
    ngqpt: list or tuple or None
        Monkhorst-Pack divisions for the phonon q-mesh.
        Default is the same as the one used in the GS calculation.
        Must be a sub-mesh of the k-mesh used for electrons.
    qptopt: int or None
        Option for the generation of the q-points list, default same as kptopt in gs.
    """

    name: str = "DFPT"
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(factory=scf_for_phonons)
        )
    )
    ddk_maker: BaseAbinitMaker | None = field(default_factory=DdkMaker)  # |
    dde_maker: BaseAbinitMaker | None = field(default_factory=DdeMaker)  # |
    dte_maker: BaseAbinitMaker | None = field(default_factory=DteMaker)  # |
    wfq_maker: BaseAbinitMaker | None = None  # | not implemented
    phonon_maker: BaseAbinitMaker | None = field(
        default_factory=PhononResponseMaker
    )  # |
    mrgddb_maker: Maker | None = field(default_factory=MrgddbMaker)  # |
    mrgdv_maker: Maker | None = field(default_factory=MrgdvMaker)  # |
    anaddb_maker: Maker | None = field(default_factory=AnaddbMaker)  # |
    use_dde_sym: bool = True
    dte_skip_permutations: bool | None = False
    qpt_list: list[list] | None = None
    ngqpt: list | None = None
    qptopt: int = None

    def __post_init__(self) -> None:
        """Process post-init configuration."""
        if self.dde_maker and not self.ddk_maker:
            raise ValueError(
                "DDK calculations are required to continue \
                with the DDE calculations. Either provide a DDK Maker \
                or remove the DDE one."
            )
        if self.dte_maker and not self.dde_maker:
            raise ValueError(
                "DDE calculations are required to continue \
                with the DTE calculations. Either provide a DDE Maker \
                or remove the DTE one."
            )
        if self.dte_maker and self.use_dde_sym:
            raise ValueError(
                "DTE calculations require all the DDE perturbations, \
                the use of symmetries is not allowed."
            )
        if self.anaddb_maker and not self.mrgddb_maker:
            raise ValueError(
                "Anaddb should be used to analyze a merged DDB. \
                Either provide a Mrgddb Maker \
                or remove the AnaddbMaker."
            )

    def make(
        self,
        structure: Structure | None = None,
        restart_from: str | Path | None = None,
        **anaddb_kwargs,
    ) -> Flow:
        """
        Create a DFPT flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.
        anaddb_kwargs : dict
            Additional kwargs for the anaddb maker.

        Returns
        -------
        Flow
            A DFPT flow
        """
        jobs = []
        if (
            isinstance(self.static_maker, StaticMakerforPhonons)
            and not self.wfq_maker
            and self.ngqpt
        ):
            """A check on the q-mesh is performed in order to avoid gs computations
            if q and k grids are not commensurate."""

            static_job = self.static_maker.validate_grids(
                structure, ngqpt=self.ngqpt, restart_from=restart_from
            )
            static_job.name = "SCF with grids validation"
            jobs.append(static_job)

        else:
            static_job = self.static_maker.make(
                structure=structure, restart_from=restart_from
            )
            jobs.append(static_job)

        if self.ddk_maker:
            # the use of symmetries is not implemented for DDK
            perturbations = [{"idir": 1}, {"idir": 2}, {"idir": 3}]
            ddk_jobs = []
            outputs: dict[str, list] = {"dirs": []}
            for ipert, pert in enumerate(perturbations):
                ddk_job = self.ddk_maker.make(
                    perturbation=pert,
                    prev_outputs=static_job.output.dir_name,
                )
                ddk_job.append_name(f"{ipert+1}/{len(perturbations)}")

                ddk_jobs.append(ddk_job)
                outputs["dirs"].append(ddk_job.output.dir_name)

            ddk_calcs = Flow(ddk_jobs, outputs)
            jobs.append(ddk_calcs)

        if self.dde_maker:
            # generate the perturbations for the DDE calculations
            dde_perts = generate_dde_perts(
                gsinput=static_job.output.input.abinit_input,
                use_symmetries=self.use_dde_sym,
            )
            jobs.append(dde_perts)

            # perform the DDE calculations
            dde_calcs = run_rf(
                perturbations=dde_perts.output["perts"],
                rf_maker=self.dde_maker,
                prev_outputs=[static_job.output.dir_name, ddk_calcs.output["dirs"]],
            )
            jobs.append(dde_calcs)

        if self.dte_maker:
            phonon_pert = False

            # To uncomment once there is a PhononMaker or something similar
            # if self.ph_maker:
            #     phonon_pert = True

            # generate the perturbations for the DTE calculations
            dte_perts = generate_dte_perts(
                gsinput=static_job.output.input.abinit_input,
                skip_permutations=self.dte_skip_permutations,
                phonon_pert=phonon_pert,
            )
            jobs.append(dte_perts)

            # perform the DTE calculations
            dte_calcs = run_rf(
                perturbations=dte_perts.output["perts"],
                rf_maker=self.dte_maker,
                prev_outputs=[
                    static_job.output.dir_name,
                    dde_calcs.output["dirs"],
                ],
            )
            jobs.append(dte_calcs)

        if self.phonon_maker:
            prev_outputs = [static_job.output.dir_name]
            # if self.dde_maker:
            #    prev_outputs.append(dde_calcs.output["dirs"])
            # generation of qpt_list (if needed) and corresponding perturbations
            phonon_perts_qpt_list = generate_phonon_perts(
                gsinput=static_job.output.input.abinit_input,
                ngqpt=self.ngqpt,
                qptopt=self.qptopt,
                qpt_list=self.qpt_list,
            )
            jobs.append(phonon_perts_qpt_list)
            # perform the phonon calculations
            phonon_calcs = run_rf(
                perturbations=phonon_perts_qpt_list.output["perts"],
                rf_maker=self.phonon_maker,
                prev_outputs=prev_outputs,
                with_dde=bool(self.dde_maker),
            )
            jobs.append(phonon_calcs)

        if self.mrgddb_maker:
            # merge the DDE, DTE and Phonon DDB.
            prev_outputs = []
            if self.dde_maker:
                prev_outputs.append(dde_calcs.output["dirs"])
            if self.dte_maker:
                prev_outputs.append(dte_calcs.output["dirs"])
            if self.phonon_maker:
                prev_outputs.append(phonon_calcs.output["dirs"])

            mrgddb_job = self.mrgddb_maker.make(
                prev_outputs=prev_outputs,
            )

            jobs.append(mrgddb_job)

        if self.mrgdv_maker:
            # merge the DDE and Phonon POT files.
            prev_outputs = []
            if self.dde_maker:
                prev_outputs.append(dde_calcs.output["dirs"])
            if self.phonon_maker:
                prev_outputs.append(phonon_calcs.output["dirs"])

            mrgdv_job = self.mrgdv_maker.make(
                prev_outputs=prev_outputs,
            )
            jobs.append(mrgdv_job)

        if self.anaddb_maker:
            # analyze a merged DDB.
            if self.phonon_maker:
                # set the required args for the anaddb phbstdos input
                if anaddb_kwargs:
                    anaddb_kwargs.update(
                        {"ngqpt": phonon_perts_qpt_list.output["ngqpt"]}
                    )
                else:
                    anaddb_kwargs = {"ngqpt": phonon_perts_qpt_list.output["ngqpt"]}
                anaddb_kwargs.setdefault("nqsmall", 10)
                # ifc needed to create the phonopy like outdoc, user can turn it off
                anaddb_kwargs.setdefault("with_ifc", True)

            anaddb_job = self.anaddb_maker.make(
                structure=mrgddb_job.output.structure,
                prev_outputs=mrgddb_job.output.dir_name,
                **anaddb_kwargs,
            )

            jobs.append(anaddb_job)

        return Flow(
            jobs, output=[j.output for j in jobs], name=self.name
        )  # TODO: fix outputs


@dataclass
class ShgFlowMaker(DfptFlowMaker):
    """
    Maker to compute the static DFPT second-harmonic generation tensor.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    scissor: float
        A rigid shift of the conduction bands in eV.
    """

    name: str = "DFPT Chi2 SHG"
    anaddb_maker: Maker | None = field(default_factory=AnaddbDfptDteMaker)
    phonon_maker: Maker | None = None
    mrgdv_maker: Maker | None = None
    use_dde_sym: bool = False
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMaker(input_set_generator=ShgStaticSetGenerator())
    )
    scissor: float | None = None

    def make(
        self,
        structure: Structure | None = None,
        restart_from: str | Path | None = None,
        **anaddb_kwargs,
    ) -> Flow:
        """
        Create a DFPT flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.

        Returns
        -------
        Flow
            A DFPT flow
        """
        shg_flow = super().make(
            structure=structure, restart_from=restart_from, **anaddb_kwargs
        )

        if self.scissor:
            shg_flow = update_user_abinit_settings(
                shg_flow,
                {"dfpt_sciss": self.scissor * abu.eV_Ha},
                name_filter="Scf calculation",
            )

        return shg_flow


@dataclass
class PhononMaker(DfptFlowMaker):
    """
    Maker to generate a phonon band structure and phonon DOS flow with abinit.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    with_dde : bool
        True if the DDE calculations should be included, False otherwise.
    run_anaddb : bool
        True if the anaddb calculations should be included, False otherwise.
    run_mrgddb : bool
        True if the merge of DDB files should be included, False otherwise.
    run_mrgdv : bool
        True if the merge of POT files should be included, False otherwise.
    """

    name: str = "Phbands-PhDOS Flow"
    with_dde: bool = True
    run_anaddb: bool = True
    run_mrgddb: bool = True
    run_mrgdv: bool = True
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMakerforPhonons(
            input_set_generator=StaticSetGenerator(factory=scf_for_phonons)
        )
    )
    anaddb_maker: BaseAbinitMaker | None = field(default_factory=AnaddbPhBandsDOSMaker)
    dte_maker: BaseAbinitMaker | None = None

    def __post_init__(self) -> None:
        """Process post-init configuration."""
        if not self.with_dde:
            """
            To turn off the DDE calculations, turn off DDK as well.
            If a DDK maker is provided, it will be removed
            """
            self.ddk_maker = None
            self.dde_maker = None

        if not self.run_mrgddb:
            """Turn off the merge of POT files"""
            self.mrgddb_maker = None

        if not self.run_mrgdv:
            """Turn off the merge of DDB files"""
            self.mrgdv_maker = None

        if not self.run_anaddb:
            """Turn off the anaddb calculations"""
            self.anaddb_maker = None

    def make(
        self,
        structure: Structure | None = None,
        restart_from: str | Path | None = None,
        **anaddb_kwargs,
    ) -> Flow:
        """
        Create a phonon band structure and DOS flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.
        anaddb_kwargs : dict
            Additional kwargs for the anaddb maker.

        Returns
        -------
        Flow
            A phonon band structure and DOS flow.
        """
        return super().make(
            structure=structure,
            restart_from=restart_from,
            **anaddb_kwargs,
        )
