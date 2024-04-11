"""Module defining base mrgddb input set and generator."""

from __future__ import annotations

import copy
import logging
import os
import time
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from abipy.abio.input_tags import DDE, DTE
from abipy.flowtk.utils import Directory, irdvars_for_ext
from pymatgen.io.core import InputGenerator, InputSet

from atomate2.abinit.files import load_abinit_input
from atomate2.abinit.utils.common import (
    INDATAFILE_PREFIX,
    INDIR_NAME,
    MRGDDB_INPUT_FILE_NAME,
    OUTDATAFILE_PREFIX,
    OUTDIR_NAME,
    TMPDIR_NAME,
    InitializationError,
)

__all__ = ["MrgddbInputSet", "MrgddbInputGenerator", "MrgddbSetGenerator"]

logger = logging.getLogger(__name__)


class MrgddbInputSet(InputSet):
    """
    A class to represent a set of Mrgddb inputs.

    Parameters
    ----------
    mrgddb_input
        An input (str) to mrgddb.
    input_files
        A list of input files needed for the calculation.
    """

    def __init__(
        self,
        mrgddb_input: str = None,
        input_files: Iterable[tuple[str, str]] | None = None,
    ):
        self.input_files = input_files
        super().__init__(
            inputs={
                MRGDDB_INPUT_FILE_NAME: mrgddb_input,
            }
        )

    def write_input(
        self,
        directory: str | Path,
        make_dir: bool = True,
        overwrite: bool = True,
        zip_inputs: bool = False,
    ):
        """Write Mrgddb input file to a directory."""
        # TODO: do we allow zipping ? not sure if it really makes sense for abinit as
        #  the abinit input set also sets up links to previous files, sets up the
        #  indir, outdir and tmpdir, ...
        super().write_input(
            directory=directory,
            make_dir=make_dir,
            overwrite=overwrite,
            zip_inputs=zip_inputs,
        )
        indir, outdir, tmpdir = self.set_workdir(workdir=directory)

    def validate(self) -> bool:
        """Validate the input set.

        Check that all input files exist and are DDB files.
        """
        if not self.input_files:
            return False
        for _out_filepath, in_file in self.input_files:
            if not os.path.isfile(_out_filepath) or in_file != "in_DDB":
                return False
        return True

    @property
    def mrgddb_input(self):
        """Get the Mrgddb input (str)."""
        return self[MRGDDB_INPUT_FILE_NAME]

    @staticmethod
    def set_workdir(workdir):
        """Set up the working directory.

        This also sets up and creates standard input, output and temporary directories.
        """
        workdir = os.path.abspath(workdir)

        # Directories with input|output|temporary data.
        indir = Directory(os.path.join(workdir, INDIR_NAME))
        outdir = Directory(os.path.join(workdir, OUTDIR_NAME))
        tmpdir = Directory(os.path.join(workdir, TMPDIR_NAME))

        # Create dirs for input, output and tmp data.
        indir.makedirs()
        outdir.makedirs()
        tmpdir.makedirs()

        return indir, outdir, tmpdir

    def runlevel(self):
        """Get the set of strings defining the calculation type."""
        return self.abinit_input.runlevel

    def deepcopy(self):
        """Deep copy of the input set."""
        return copy.deepcopy(self)


PrevOutput = namedtuple("PrevOutput", "dirname exts")


@dataclass
class MrgddbInputGenerator(InputGenerator):
    """
    A class to generate Mrgddb input sets.

    Parameters
    ----------
    calc_type
        A short description of the calculation type
    restart_from_deps:
        Defines the files that needs to be linked from previous calculations in
        case of restart. The format is a tuple where each element is a list of
        "|" separated runelevels (as defined in the AbinitInput object) followed
        by a colon and a list of "|" list of extensions of files that needs to
        be linked. The runlevel defines the type of calculations from which the
        file can be linked. An example is (f"{NSCF}:WFK",).
    prev_outputs_deps
        Defines the files that needs to be linked from previous calculations and
        are required for the execution of the current calculation.
        The format is a tuple where each element is a list of  "|" separated
        runelevels (as defined in the AbinitInput object) followed by a colon and
        a list of "|" list of extensions of files that needs to be linked.
        The runlevel defines the type of calculations from which the file can
        be linked. An example is (f"{NSCF}:WFK",).
    """

    calc_type: str = "mrgddb_merge"
    restart_from_deps: str | tuple | None = None
    prev_outputs_deps: str | tuple | None = None

    def get_input_set(  # type: ignore
        self,
        restart_from: str | tuple | list | Path | None = None,
        prev_outputs: str | tuple | list | Path | None = None,
        workdir: str | Path | None = ".",
    ) -> MrgddbInputSet:
        """Generate an MrgddbInputSet object.

        Here we assume that restart_from is a directory and prev_outputs is
        a list of directories.

        Parameters
        ----------
        restart_from : str or Path or list or tuple
            Directory (as a str or Path) or list/tuple of 1 directory (as a str
            or Path) to restart from.
        prev_outputs : str or Path or list or tuple
            Directory (as a str or Path) or list/tuple of directories (as a str
            or Path) needed as dependencies for the MrgddbInputSet generated.
        """
        restart_from = self.check_format_prev_dirs(restart_from)
        prev_outputs = self.check_format_prev_dirs(prev_outputs)

        all_irdvars = {}
        input_files = []
        if restart_from is not None:
            # Use the previous mrgddb input
            mrgddb_input = load_mrgddb_input(restart_from[0])
            # Files for restart
            irdvars, files = self.resolve_deps(
                restart_from, deps=self.restart_from_deps
            )
            all_irdvars.update(irdvars)
            input_files.extend(files)
        else:
            if prev_outputs is not None and not self.prev_outputs_deps:
                raise RuntimeError(
                    f"Previous outputs not allowed for {self.__class__.__name__}."
                )
            irdvars, files = self.resolve_deps(prev_outputs, self.prev_outputs_deps)
            input_files.extend(files)
            mrgddb_input = self.get_mrgddb_input(
                prev_outputs=prev_outputs,
                workdir=workdir,
            )

        return MrgddbInputSet(
            mrgddb_input=mrgddb_input,
            input_files=input_files,
        )

    def check_format_prev_dirs(
        self, prev_dirs: str | tuple | list | Path | None
    ) -> list[str] | None:
        """Check and format the prev_dirs (restart or dependency)."""
        if prev_dirs is None:
            return None
        if isinstance(prev_dirs, (str, Path)):
            return [str(prev_dirs)]
        return [str(prev_dir) for prev_dir in prev_dirs]

    def resolve_deps(
        self, prev_dirs: list[str], deps: str | tuple, check_runlevel: bool = True
    ) -> tuple[dict, list]:
        """Resolve dependencies.

        This method assumes that prev_dirs is in the correct format, i.e.
        a list of directories as str or Path.
        """
        input_files = []
        deps_irdvars = {}
        for prev_dir in prev_dirs:
            if check_runlevel:
                abinit_input = load_abinit_input(prev_dir)
            for dep in deps:
                runlevel = set(dep.split(":")[0].split("|"))
                exts = list(dep.split(":")[1].split("|"))
                if not check_runlevel or runlevel.intersection(abinit_input.runlevel):
                    irdvars, inp_files = self.resolve_dep_exts(
                        prev_dir=prev_dir, exts=exts
                    )
                    input_files.extend(inp_files)
                    deps_irdvars.update(irdvars)

        return deps_irdvars, input_files

    @staticmethod
    def _get_in_file_name(out_filepath: str) -> str:
        in_file = os.path.basename(out_filepath)
        in_file = in_file.replace(OUTDATAFILE_PREFIX, INDATAFILE_PREFIX, 1)
        in_file = os.path.basename(in_file).replace("WFQ", "WFK", 1)
        return in_file

    @staticmethod
    def resolve_dep_exts(prev_dir: str, exts: list[str]) -> tuple:
        """Return irdvars and corresponding file for a given dependency.

        This method assumes that prev_dir is in the correct format,
        i.e. a directory as a str or Path.
        """
        prev_outdir = Directory(os.path.join(prev_dir, OUTDIR_NAME))
        inp_files = []

        for ext in exts:
            # TODO: how to check that we have the files we need ?
            #  Should we raise if don't find at least one file for a given extension ?
            if ext in ("1WF", "1DEN"):
                # Special treatment for 1WF and 1DEN files
                if ext == "1WF":
                    files = prev_outdir.find_1wf_files()
                elif ext == "1DEN":
                    files = prev_outdir.find_1den_files()
                else:
                    raise RuntimeError("Should not occur.")
                if files is not None:
                    inp_files = [
                        (f.path, MrgddbInputGenerator._get_in_file_name(f.path))
                        for f in files
                    ]
                    irdvars = irdvars_for_ext(ext)
                    break
            elif ext == "DEN":
                # Special treatment for DEN files
                # In case of relaxations or MD, there may be several TIM?_DEN files
                # First look for the standard out_DEN file.
                # If not found, look for the last TIM?_DEN file.
                out_den = prev_outdir.path_in(f"{OUTDATAFILE_PREFIX}_DEN")
                if os.path.exists(out_den):
                    irdvars = irdvars_for_ext("DEN")
                    inp_files.append(
                        (out_den, MrgddbInputGenerator._get_in_file_name(out_den))
                    )
                    break
                last_timden = prev_outdir.find_last_timden_file()
                if last_timden is not None:
                    if last_timden.path.endswith(".nc"):
                        in_file_name = f"{INDATAFILE_PREFIX}_DEN.nc"
                    else:
                        in_file_name = f"{INDATAFILE_PREFIX}_DEN"
                    inp_files.append((last_timden.path, in_file_name))
                    irdvars = irdvars_for_ext("DEN")
                    break
            else:
                out_file = prev_outdir.has_abiext(ext)
                irdvars = irdvars_for_ext(ext)
                if out_file:
                    inp_files.append(
                        (out_file, MrgddbInputGenerator._get_in_file_name(out_file))
                    )
                    break
        else:
            msg = f"Cannot find {' or '.join(exts)} file to restart from."
            logger.error(msg)
            raise InitializationError(msg)
        return irdvars, inp_files

    def get_mrgddb_input(
        self,
        prev_outputs: list[str] | None = None,
        workdir: str | Path | None = ".",
    ) -> str:
        """
        Generate the mrgddb input (str) for the input set.

        Parameters
        ----------
        prev_outputs
            A list of previous output directories.

        Returns
        -------
            A string
        """
        if not prev_outputs:
            raise RuntimeError(
                f"No previous_outputs. Required for {self.__class__.__name__}."
            )

        if not self.prev_outputs_deps and prev_outputs:
            msg = (
                f"Previous outputs not allowed for {self.__class__.__name__} "
                "Consider if restart_from argument of get_input_set method "
                "can fit your needs instead."
            )
            raise RuntimeError(msg)

        irdvars, files = self.resolve_deps(prev_outputs, self.prev_outputs_deps)

        workdir = os.path.abspath(workdir)
        outdir = Directory(os.path.join(workdir, OUTDIR_NAME, "out_DDB"))

        generated_input = str(outdir)
        generated_input += "\n"
        generated_input += f"DDBs merged on {time.asctime()}"
        generated_input += "\n"
        generated_input += f"{len(files)}"
        for file_path, file_name in files:
            generated_input += "\n"
            generated_input += f"{file_path}"

        return generated_input


@dataclass
class MrgddbSetGenerator(MrgddbInputGenerator):
    """Class to generate Mrgddb input sets."""

    calc_type: str = "mrgddb_merge"
    restart_from_deps: tuple = None  # (f"{MRGDDB}:DDB",) #TODO: name is MRGDDB ?
    prev_outputs_deps: tuple = (f"{DDE}:DDB", f"{DTE}:DDB")

    def get_mrgddb_input(
        self,
        prev_outputs: list[str] | None = None,
        workdir: str | Path | None = ".",
    ) -> str:
        """Get Mrgddb input (str) to merge the DDE and DTE DDB."""
        return super().get_mrgddb_input(
            prev_outputs=prev_outputs,
        )
