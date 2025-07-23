"""Merge DV jobs for merging POT files from ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jobflow
import numpy as np
from jobflow import Maker, Response, job

from atomate2.abinit.files import write_mrgdv_input_set
from atomate2.abinit.jobs.base import setup_job
from atomate2.abinit.run import run_mrgdv
from atomate2.abinit.schemas.mrgdvdb import MrgdvdbTaskDoc
from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.sets.mrgdvdb import MrgdvInputGenerator

if TYPE_CHECKING:
    from collections.abc import Callable

    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "MrgdvMaker",
]

_MRGDV_DATA_OBJECTS = [
    AbinitStoredFile,
]


def mrgdv_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of mrgdv job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.job` that configures common
    settings for all mrgdv jobs. For example, it ensures that the mrgdvdb file is
    stored in the atomate2 data store. It also configures the output schema to be a
    mrgdvdb
    :obj:`.TaskDocument`.

    Any makers that return mrgdv jobs (not flows) should decorate the ``make`` method
    with @mrgdv_job. For example:

    .. code-block:: python

        class MyMrgdvMaker(MrgdvMaker):
            @mrgdv_job
            def make(structure):
                # code to run mrgdv job.
                pass

    Parameters
    ----------
    method : callable
        A BaseMrgdvMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate mrgdv jobs.
    """
    return job(method, data=_MRGDV_DATA_OBJECTS, output_schema=MrgdvdbTaskDoc)


@dataclass
class MrgdvMaker(Maker):
    """Maker to create a job with a merge of POT files from ABINIT.

    Parameters
    ----------
    name : str
        The job name.
    """

    name: str = "Merge DVDB"
    input_set_generator: MrgdvInputGenerator = field(
        default_factory=MrgdvInputGenerator
    )
    wall_time: int | None = None

    @property
    def calc_type(self) -> str:
        """Get the type of calculation for this maker."""
        return self.input_set_generator.calc_type

    @mrgdv_job
    def make(
        self,
        prev_outputs: list[str] | None = None,
        history: JobHistory | None = None,
    ) -> jobflow.Response:
        """
        Return a MRGdv jobflow.Job.

        Parameters
        ----------
        prev_outputs : TODO: add description from sets.base
        history : JobHistory
            A JobHistory object containing the history of this job.
        """
        # Flatten the list of previous outputs dir
        # prev_outputs = [item for sublist in prev_outputs for item in sublist]
        prev_outputs = list(np.hstack(prev_outputs))

        # Setup job and get general job configuration
        config = setup_job(
            structure=None,
            prev_outputs=prev_outputs,
            restart_from=None,
            history=history,
            wall_time=self.wall_time,
        )

        # Write mrgdv input set
        write_mrgdv_input_set(
            input_set_generator=self.input_set_generator,
            prev_outputs=prev_outputs,
            directory=config.workdir,
        )

        # Run mrgdv
        run_mrgdv(
            wall_time=config.wall_time,
            start_time=config.start_time,
        )

        # parse Mrgdv dv output
        task_doc = MrgdvdbTaskDoc.from_directory(
            Path.cwd(),
            # **self.task_document_kwargs,
        )
        task_doc.task_label = self.name

        return Response(
            output=task_doc,
        )
