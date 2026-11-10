from __future__ import annotations

import gzip
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from abipy.abio.abivars import is_anaddb_var
from pymatgen.util.coord import find_in_coord_list_pbc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core.structure import Structure
    from pytest import TempPathFactory


logger = logging.getLogger("atomate2")

_REF_PATHS = {}
_ABINIT_FILES = ("run.abi", "abinit_input.json")
_FAKE_RUN_ABINIT_KWARGS = {}
# Reference paths for the abipy perturbation helpers mocked by mock_abinit,
# resolved through the standard _REF_PATHS registry: the reference of the
# generate_perts job is the perts.json.gz file inside its reference directory.
# Results are stored under a canonical string of the distinguishing call
# arguments (see _pert_call_key).
# Perturbation reference files already (re-)written during the current fixture:
# the first record-mode call resets the file, so re-recording a test does not
# append duplicate entries to existing reference data.
_PERT_RECORDED_FILES = set()
# The abipy AbinitInput methods mocked by mock_abinit. These helpers would
# otherwise run the real abinit binary from within the generate_perts job.
_PERT_METHODS = (
    "abiget_irred_ddeperts",
    "abiget_irred_dteperts",
    "abiget_irred_phperts",
    "abiget_ibz",
)
_MRGDDB_FILES = "mrgddb.in"
_FAKE_RUN_MRGDDB_KWARGS = {}
_MRGDV_FILES = "mrgdv.in"
_FAKE_RUN_MRGDV_KWARGS = {}
_ANADDB_FILES = ("anaddb.in", "anaddb_input.json")
_FAKE_RUN_ANADDB_KWARGS = {}

# Do this here to prevent issues with threaded CI runners
# In abipy, it's possible to have thread collisions in
# making this directory because `exist_ok = False` there
_ABINIT_PATH = Path("~/.abinit/abipy").expanduser()
if not _ABINIT_PATH.is_dir():
    _ABINIT_PATH.mkdir(exist_ok=True, parents=True)


@pytest.fixture(scope="session")
def abinit_test_dir(test_dir: Path) -> Path:
    """
    Get the ABINIT test directory.

    Parameters
    ----------
    test_dir
        The root test directory.

    Returns
    -------
    Path
        The ABINIT test directory path.
    """
    return test_dir / "abinit"


@pytest.fixture(scope="session", autouse=True)
def load_pseudos(abinit_test_dir: Path, tmp_path_factory: TempPathFactory):
    """
    Configure the pseudopotential repository root directory for tests.

    Parameters
    ----------
    abinit_test_dir
        The ABINIT test directory containing the pseudos subdirectory.
    """
    # Create a session-scoped temp directory
    temp_dir = tmp_path_factory.mktemp("pseudos")
    temp_dir = Path(temp_dir)

    # Path to the compressed pseudos
    compressed_pseudos_dir = abinit_test_dir / "pseudos"

    # Iterate over all .gz files in subdirectories
    for gz_file in compressed_pseudos_dir.glob("**/*.gz"):
        # Recreate the relative path in the temp directory
        relative_path = gz_file.relative_to(compressed_pseudos_dir)
        target_dir = temp_dir / relative_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        # Uncompress the file to the target directory
        target_file = target_dir / relative_path.stem
        with gzip.open(gz_file, "rb") as f_in, open(target_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Set the REPOS_ROOT to the temp directory
    import abipy.flowtk.psrepos

    abipy.flowtk.psrepos.REPOS_ROOT = str(temp_dir)
    # Cleanup is automatic for tmp_path_factory


@pytest.fixture(scope="session", autouse=True)
def load_manager(abinit_test_dir: Path) -> None:
    """
    Configure the ABINIT task manager configuration directory for tests.

    Parameters
    ----------
    abinit_test_dir
        The ABINIT test directory containing the abipy subdirectory.
    """
    import abipy.flowtk.tasks

    abipy.flowtk.tasks.TaskManager.USER_CONFIG_DIR = str(abinit_test_dir / "abipy")


@pytest.fixture(scope="session")
def abinit_integration_tests(pytestconfig: pytest.Config) -> bool:
    """
    Get the ABINIT integration test flag from pytest configuration.

    Parameters
    ----------
    pytestconfig
        The pytest configuration object.

    Returns
    -------
    bool
        True if ABINIT integration tests are enabled, False otherwise.
    """
    return pytestconfig.getoption("abinit_integration")


@pytest.fixture(scope="session")
def abinit_record_perts(pytestconfig: pytest.Config) -> bool:
    """
    Get the record perturbations flag from pytest configuration.

    Parameters
    ----------
    pytestconfig
        The pytest configuration object.

    Returns
    -------
    bool
        True if abipy perturbation results should be recorded to the
        perts.json.gz reference files, False otherwise.
    """
    return pytestconfig.getoption("abinit_record_perts")


@pytest.fixture
def mock_abinit(
    monkeypatch, abinit_test_dir, abinit_integration_tests, abinit_record_perts
):
    """
    This fixture allows one to mock running ABINIT.

    It works by monkeypatching (replacing) calls to run_abinit.

    The primary idea is that instead of running ABINIT to generate the output files,
    reference files will be copied into the directory instead.
    """
    import atomate2.abinit.files
    import atomate2.abinit.jobs.base
    import atomate2.abinit.run

    # Wrap write_abinit_input_set to check inputs after writing
    def wrapped_write_abinit_input_set(*args, **kwargs) -> None:
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        index = CURRENT_JOB.job.index
        ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]

        atomate2.abinit.files.write_abinit_input_set(*args, **kwargs)
        check_abinit_inputs(ref_path)

    monkeypatch.setattr(
        atomate2.abinit.jobs.base,
        "write_abinit_input_set",
        wrapped_write_abinit_input_set,
    )

    if not abinit_integration_tests:
        # Mock ABINIT run by copying reference output files
        def mock_run_abinit(
            wall_time: float | None = None, start_time: float | None = None
        ) -> None:
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_abinit_inputs(ref_path)
            fake_run_abinit(ref_path)

        monkeypatch.setattr(atomate2.abinit.run, "run_abinit", mock_run_abinit)
        monkeypatch.setattr(atomate2.abinit.jobs.base, "run_abinit", mock_run_abinit)

        # Mock the abipy perturbation helpers used by the generate_perts job in
        # order to prevent them from running the real abinit binary: their
        # results are replayed from the perts.json.gz reference file of the
        # generate_perts job, resolved through _REF_PATHS. Passing
        # --abinit-record-perts activates record mode: the real abipy methods
        # are then called (this requires a working `abinit` binary on PATH) and
        # their results are recorded to the perts.json.gz reference file.
        from abipy.abio.inputs import AbinitInput

        record_perts = abinit_record_perts
        if record_perts:
            _PERT_RECORDED_FILES.clear()

        def _make_fake_pert_method(method_name, orig_method, record=False):
            def _fake(self, *args, workdir=None, **kwargs):
                from jobflow import CURRENT_JOB

                name = CURRENT_JOB.job.name  # expected: "generate_perts"
                index = CURRENT_JOB.job.index
                if name not in _REF_PATHS or not _REF_PATHS[name]:
                    raise ValueError(
                        f"No reference path registered for job {name} - {index}."
                    )
                ref_dirs = _REF_PATHS[name]
                if str(index) in ref_dirs:
                    ref_dir = abinit_test_dir / ref_dirs[str(index)]
                elif len(ref_dirs) == 1:
                    # The index under which the CLI registers generate_perts is
                    # an artifact of jobflow's store_inputs mechanism: flow
                    # outputs are stored at the replaced job's index + 1, while
                    # the perturbation calls happen at the execution index. With
                    # a single registered ref dir there is no ambiguity.
                    ref_dir = abinit_test_dir / next(iter(ref_dirs.values()))
                else:
                    raise ValueError(
                        f"Ambiguous reference paths registered for job "
                        f"{name} - {index}: {sorted(ref_dirs)}."
                    )
                ref_file = ref_dir / "perts.json.gz"
                # Canonical key of the call arguments that distinguish the
                # call. workdir and manager never take part in the key as
                # they only define where the real abinit task runs.
                key = _pert_call_key(method_name, args, kwargs)
                if record:
                    result = orig_method(self, *args, workdir=workdir, **kwargs)
                    result_data = _dump_pert_result(method_name, result)
                    if str(ref_file) in _PERT_RECORDED_FILES and ref_file.exists():
                        data = _load_perts_gz(ref_file)
                    else:
                        # First record call for this fixture: reset the file.
                        data = {}
                    if key in data.get(method_name, {}):
                        # Duplicate call with the same distinguishing arguments:
                        # legitimate only if the abipy method is deterministic.
                        # Identical results are recorded once, different results
                        # are a loud error instead of a silent overwrite.
                        if not _pert_results_equal(
                            method_name, result_data, data[method_name][key]
                        ):
                            raise ValueError(
                                f"Inconsistent duplicate call to {method_name} "
                                f"with key {key!r}: recorded "
                                f"{data[method_name][key]} differs from the "
                                f"newly computed {result_data}."
                            )
                        logger.info(
                            f"duplicate call to {method_name} with key {key}, "
                            "identical result - recorded once"
                        )
                        return result
                    data.setdefault(method_name, {})[key] = result_data
                    _dump_perts_gz(data, ref_file)
                    _PERT_RECORDED_FILES.add(str(ref_file))
                    return result
                data = _load_perts_gz(ref_file)
                if key not in data.get(method_name, {}):
                    raise ValueError(
                        f"No recorded result for {method_name} with key "
                        f"{key!r}; recorded keys: "
                        f"{sorted(data.get(method_name, {}))}. "
                        "Re-record with --abinit-record-perts."
                    )
                return _load_pert_result(method_name, data[method_name][key])

            return _fake

        for method_name in _PERT_METHODS:
            monkeypatch.setattr(
                AbinitInput,
                method_name,
                _make_fake_pert_method(
                    method_name,
                    getattr(AbinitInput, method_name),
                    record=record_perts,
                ),
            )

    def _run(ref_paths: dict, fake_run_abinit_kwargs: dict | None = None) -> None:
        if fake_run_abinit_kwargs is None:
            fake_run_abinit_kwargs = {}
        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_ABINIT_KWARGS.update(fake_run_abinit_kwargs)

    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_ABINIT_KWARGS.clear()
    _PERT_RECORDED_FILES.clear()


def _dump_perts_gz(data: dict, filename: Path) -> None:
    """
    Write pert ref data as gzip JSON with a zeroed header (deterministic bytes).

    Python's gzip writer embeds the current mtime in the gzip header, so two
    record runs with identical data would otherwise produce different bytes.
    Together with sort_keys=True this makes two record runs byte-identical.
    """
    payload = json.dumps(data, sort_keys=True).encode()
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_bytes(gzip.compress(payload, mtime=0))


def _load_perts_gz(filename: Path) -> dict:
    """
    Read pert ref data written by _dump_perts_gz.
    """
    return json.loads(gzip.decompress(filename.read_bytes()))


def _pert_canon(value):
    """
    Convert numpy values to plain Python objects with a version-stable repr.

    Plain conversion is required so that reference keys are identical across
    numpy versions (repr of numpy scalars changed with numpy 2.0).
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pert_call_key(method_name: str, args: tuple, kwargs: dict) -> str:
    """
    Get the canonical key of the call arguments that distinguish a perturbation
    call. workdir and manager never take part in the key, as they only define
    where the real abinit task runs.

    The per-method keys are:
    - "abiget_irred_phperts": the q-point, e.g. "(0.5, 0.0, 0.0)"
    - "abiget_ibz": comma-joined "name=value" pairs of the call kwargs,
      sorted by name, e.g. "kptopt=3, shiftk=(0, 0, 0)"
    - "abiget_irred_ddeperts" and "abiget_irred_dteperts": "default" (called
      once, without distinguishing arguments)

    Parameters
    ----------
    method_name
        Name of the abipy AbinitInput method being called.
    args
        Positional call arguments (qpt can be passed positionally to
        abiget_irred_phperts).
    kwargs
        Keyword call arguments.
    """
    if method_name == "abiget_irred_phperts":
        qpt = kwargs.get("qpt", args[0] if args else None)
        if qpt is None:
            raise ValueError(
                "The q-point argument is required to compute the reference "
                f"key of {method_name}."
            )
        return str(tuple(_pert_canon(q) for q in qpt))
    if method_name == "abiget_ibz":
        return ", ".join(
            f"{k}={_pert_canon(kwargs[k])}"
            for k in sorted(kwargs)
            if k not in ("workdir", "manager")
        )
    return "default"


def _pert_results_equal(method_name: str, a, b) -> bool:
    """
    Compare two canonically serialized perturbation results for equality.

    Parameters
    ----------
    method_name
        Name of the abipy AbinitInput method that produced the results.
    a, b
        Serialized results in the reference file format.
    """
    if method_name == "abiget_ibz":
        return np.array_equal(
            np.asarray(a["points"]), np.asarray(b["points"])
        ) and np.array_equal(np.asarray(a["weights"]), np.asarray(b["weights"]))
    import json

    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _dump_pert_result(method_name: str, result) -> object:
    """
    Convert the return value of an abipy perturbation helper to JSON-safe data.

    The load counterpart _load_pert_result mirrors these transformations.

    Parameters
    ----------
    method_name
        Name of the abipy AbinitInput method that produced the result.
    result
        Return value of the abipy method.
    """
    if method_name == "abiget_ibz":
        # abiget_ibz returns a namedtuple of numpy arrays: store the arrays as
        # plain lists, the load path rebuilds the namedtuple from them.
        return {
            "points": np.asarray(result.points).tolist(),
            "weights": np.asarray(result.weights).tolist(),
        }
    # The abiget_irred_*perts methods return lists of monty AttrDict (a dict
    # subclass): store them as plain dicts, the load path restores them as
    # AttrDicts so that attribute access (e.g. pert.i1pert) keeps working.
    return [dict(pert) for pert in result]


def _load_pert_result(method_name: str, data):
    """
    Rebuild the return value of an abipy perturbation helper from reference data.

    Parameters
    ----------
    method_name
        Name of the abipy AbinitInput method that produced the result.
    data
        JSON-safe data produced by _dump_pert_result.
    """
    if method_name == "abiget_ibz":
        from typing import NamedTuple

        class Ibz(NamedTuple):
            points: np.ndarray
            weights: np.ndarray

        return Ibz(points=np.array(data["points"]), weights=np.array(data["weights"]))

    from monty.collections import AttrDict

    return [AttrDict(**pert) for pert in data]


def fake_run_abinit(ref_path: str | Path) -> None:
    """
    Emulate running ABINIT.

    Parameters
    ----------
    ref_path
        Path to reference directory with ABINIT input files in the folder named 'inputs'
        and output files in the folder named 'outputs'.
    """
    logger.info("Running fake ABINIT.")

    ref_path = Path(ref_path)

    copy_abinit_outputs(ref_path)

    # Pretend to run ABINIT by copying pre-generated outputs from reference directory
    logger.info("Generated fake ABINIT outputs")


def check_abinit_inputs(
    ref_path: str | Path,
    check_inputs: Sequence[Literal["run.abi"]] = _ABINIT_FILES,
) -> None:
    """
    Verify ABINIT input files against reference inputs.

    Parameters
    ----------
    ref_path
        Path to reference directory containing input files.
    check_inputs
        Sequence of input file names to check.
    """
    ref_path = Path(ref_path)

    if "run.abi" in check_inputs:
        check_run_abi(ref_path)

    if "abinit_input.json" in check_inputs:
        check_abinit_input_json(ref_path)

    logger.info("Verified inputs successfully")


def check_run_abi(ref_path: str | Path) -> None:
    """
    Verify run.abi input file against reference.

    Parameters
    ----------
    ref_path
        Path to reference directory containing the reference run.abi file.
    """
    from abipy.abio.abivars import AbinitInputFile
    from monty.io import zopen

    user = AbinitInputFile.from_file("run.abi")
    assert user.ndtset == 1, f"'run.abi' has multiple datasets (ndtset={user.ndtset})."
    with zopen(ref_path / "inputs" / "run.abi.gz", "rt", encoding="utf-8") as file:
        ref_str = file.read()
    ref = AbinitInputFile.from_string(ref_str)
    # Ignore pseudos and pp_dirpath as they depend on the pseudo root directory
    diffs = user.get_differences(ref, ignore_vars=["pseudos", "pp_dirpath"])
    # TODO: Should we still add some check on the pseudos here?
    assert diffs == [], f"'run.abi' is different from reference: \n{diffs}"


# Adapted from check_poscar in atomate2.utils.testing.vasp.py
def check_equivalent_frac_coords(
    struct: Structure,
    struct_ref: Structure,
    atol: float = 1e-3,
) -> None:
    """
    Check that the fractional coordinates of two structures are equivalent.

    This function verifies that all sites in the structure match the reference
    structure's sites within a tolerance, accounting for periodic boundary conditions
    (PBC). Site ordering is not required to match.

    Parameters
    ----------
    struct
        The structure to check.
    struct_ref
        The reference structure.
    atol
        Absolute tolerance for coordinate comparison.
    """
    user_frac_coords = struct.frac_coords
    ref_frac_coords = struct_ref.frac_coords

    # In some cases, the ordering of sites can change when copying input files.
    # To account for this, we check that the sites are the same, within a tolerance,
    # while accounting for periodic boundary conditions.
    coord_match = [
        len(find_in_coord_list_pbc(ref_frac_coords, coord, atol=atol)) > 0
        for coord in user_frac_coords
    ]
    assert all(coord_match), (
        f"The two structures have different frac. coords: "
        f"{user_frac_coords} vs. {ref_frac_coords}."
    )


def check_equivalent_znucl_typat(
    znucl_a: list | np.ndarray,
    znucl_b: list | np.ndarray,
    typat_a: list | np.ndarray,
    typat_b: list | np.ndarray,
) -> None:
    """
    Check that the elements and their counts are equivalent between two structures.

    Parameters
    ----------
    znucl_a
        Atomic numbers from the first structure.
    znucl_b
        Atomic numbers from the second structure.
    typat_a
        Atom types from the first structure.
    typat_b
        Atom types from the second structure.
    """
    sorted_znucl_a = sorted(znucl_a, reverse=True)
    sorted_znucl_b = sorted(znucl_b, reverse=True)
    assert sorted_znucl_a == sorted_znucl_b, (
        f"The elements are different: {znucl_a} vs. {znucl_b}"
    )

    count_sorted_znucl_a = [
        list(typat_a).count(list(znucl_a).index(s) + 1) for s in sorted_znucl_a
    ]
    count_sorted_znucl_b = [
        list(typat_b).count(list(znucl_b).index(s) + 1) for s in sorted_znucl_b
    ]
    assert count_sorted_znucl_a == count_sorted_znucl_b, (
        f"The number of same elements is different: "
        f"{count_sorted_znucl_a} vs. {count_sorted_znucl_b}"
    )


def check_abinit_input_json(ref_path: str | Path) -> None:
    """
    Verify abinit_input.json against reference.

    Parameters
    ----------
    ref_path
        Path to reference directory containing the reference abinit_input.json file.
    """
    from abipy.abio.inputs import AbinitInput
    from monty.serialization import loadfn

    user = loadfn("abinit_input.json")
    assert isinstance(user, AbinitInput)
    user_abivars = user.structure.to_abivars()

    ref = loadfn(ref_path / "inputs" / "abinit_input.json.gz")
    ref_abivars = ref.structure.to_abivars()

    check_equivalent_frac_coords(user.structure, ref.structure)
    check_equivalent_znucl_typat(
        user_abivars["znucl"],
        ref_abivars["znucl"],
        user_abivars["typat"],
        ref_abivars["typat"],
    )

    for k, user_v in user_abivars.items():
        if k in ["xred", "znucl", "typat"]:
            continue
        assert k in ref_abivars, f"{k = } is not a key of the reference input."
        ref_v = ref_abivars[k]
        if isinstance(user_v, str):
            assert user_v == ref_v, f"{k = }-->{user_v = } versus {ref_v = }"
        else:
            assert np.allclose(user_v, ref_v), f"{k = }-->{user_v = } versus {ref_v = }"
    assert user.runlevel == ref.runlevel, f"{user.runlevel = } versus {ref.runlevel = }"


def clear_abinit_files() -> None:
    """Remove ABINIT output files from the current directory."""
    for abinit_file in ("run.abo",):
        if Path(abinit_file).exists():
            Path(abinit_file).unlink()
    logger.info("Cleared abinit files.")


@pytest.fixture
def mock_mrgddb(mocker, abinit_test_dir, abinit_integration_tests):
    """
    This fixture allows one to mock running Mrgddb.

    It works by monkeypatching (replacing) calls to run_mrgddb.

    The primary idea is that instead of running Mrgddb to generate the output files,
    reference files will be copied into the directory instead.
    """
    import atomate2.abinit.files
    import atomate2.abinit.jobs.mrgddb
    import atomate2.abinit.run

    # Wrap write_mrgddb_input_set to check inputs after writing
    def wrapped_write_mrgddb_input_set(*args, **kwargs) -> None:
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        index = CURRENT_JOB.job.index
        ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]

        atomate2.abinit.files.write_mrgddb_input_set(*args, **kwargs)
        check_mrgddb_inputs(ref_path)

    mocker.patch.object(
        atomate2.abinit.jobs.mrgddb,
        "write_mrgddb_input_set",
        wrapped_write_mrgddb_input_set,
    )

    if not abinit_integration_tests:
        # Mock mrgddb run by copying reference output files
        def mock_run_mrgddb(
            wall_time: float | None = None, start_time: float | None = None
        ) -> None:
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_mrgddb_inputs(ref_path)
            fake_run_abinit(ref_path)

        mocker.patch.object(atomate2.abinit.run, "run_mrgddb", mock_run_mrgddb)
        mocker.patch.object(atomate2.abinit.jobs.mrgddb, "run_mrgddb", mock_run_mrgddb)

    def _run(ref_paths: dict, fake_run_mrgddb_kwargs: dict | None = None) -> None:
        if fake_run_mrgddb_kwargs is None:
            fake_run_mrgddb_kwargs = {}
        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_MRGDDB_KWARGS.update(fake_run_mrgddb_kwargs)

    yield _run

    mocker.stopall()
    _REF_PATHS.clear()
    _FAKE_RUN_MRGDDB_KWARGS.clear()


def check_mrgddb_inputs(
    ref_path: str | Path,
    check_inputs: Sequence[Literal["mrgddb.in"]] = _MRGDDB_FILES,
) -> None:
    """
    Verify mrgddb input files against reference inputs.

    Parameters
    ----------
    ref_path
        Path to reference directory containing input files.
    check_inputs
        Sequence of input file names to check.
    """
    ref_path = Path(ref_path)

    if "mrgddb.in" in check_inputs:
        from monty.io import zopen

        with open("mrgddb.in") as file:
            str_in = file.readlines()
        str_in.pop(1)  # Remove second line (DDB output path)

        with zopen(
            ref_path / "inputs" / "mrgddb.in.gz", "rt", encoding="utf-8"
        ) as file:
            ref_str = file.readlines()
        ref_str.pop(1)  # Remove second line (DDB output path)

        assert str_in[1] == ref_str[1], "'mrgddb.in' is different from reference."

        str_in.pop(1)
        ref_str.pop(1)

        # Only keep the "outdata/out_DDB\n" from the path for comparison
        for i, _ in enumerate(str_in):
            str_in[i] = str_in[i][-16:]
            ref_str[i] = ref_str[i][-16:]

        assert str_in == ref_str, "'mrgddb.in' is different from reference."

    logger.info("Verified inputs successfully")


@pytest.fixture
def mock_mrgdvdb(mocker, abinit_test_dir, abinit_integration_tests):
    """
    This fixture allows one to mock running Mrgdvdb.

    It works by monkeypatching (replacing) calls to run_mrgdv.

    The primary idea is that instead of running Mrgdvdb to generate the output files,
    reference files will be copied into the directory instead.
    """
    import atomate2.abinit.files
    import atomate2.abinit.jobs.mrgdv
    import atomate2.abinit.run

    # Wrap write_mrgdv_input_set to check inputs after writing
    def wrapped_write_mrgdv_input_set(*args, **kwargs) -> None:
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        index = CURRENT_JOB.job.index
        ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]

        atomate2.abinit.files.write_mrgdv_input_set(*args, **kwargs)
        check_mrgdv_inputs(ref_path)

    mocker.patch.object(
        atomate2.abinit.jobs.mrgdv,
        "write_mrgdv_input_set",
        wrapped_write_mrgdv_input_set,
    )

    if not abinit_integration_tests:
        # Mock mrgdv run by copying reference output files
        def mock_run_mrgdv(
            wall_time: float | None = None, start_time: float | None = None
        ) -> None:
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_mrgdv_inputs(ref_path)
            fake_run_abinit(ref_path)

        mocker.patch.object(atomate2.abinit.run, "run_mrgdv", mock_run_mrgdv)
        mocker.patch.object(atomate2.abinit.jobs.mrgdv, "run_mrgdv", mock_run_mrgdv)

    def _run(ref_paths: dict, fake_run_mrgdv_kwargs: dict | None = None) -> None:
        if fake_run_mrgdv_kwargs is None:
            fake_run_mrgdv_kwargs = {}
        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_MRGDV_KWARGS.update(fake_run_mrgdv_kwargs)

    yield _run

    mocker.stopall()
    _REF_PATHS.clear()
    _FAKE_RUN_MRGDV_KWARGS.clear()


def check_mrgdv_inputs(
    ref_path: str | Path,
    check_inputs: Sequence[Literal["mrgdv.in"]] = _MRGDV_FILES,
) -> None:
    """
    Verify mrgdv input files against reference inputs.

    Parameters
    ----------
    ref_path
        Path to reference directory containing input files.
    check_inputs
        Sequence of input file names to check.
    """
    ref_path = Path(ref_path)

    if "mrgdv.in" in check_inputs:
        from monty.io import zopen

        with open("mrgdv.in") as file:
            str_in = file.readlines()
        str_in.pop(1)  # Remove second line (POT output path)

        with zopen(ref_path / "inputs" / "mrgdv.in.gz", "rt", encoding="utf-8") as file:
            ref_str = file.readlines()
        ref_str.pop(1)  # Remove second line (POT output path)

        assert str_in[1] == ref_str[1], "'mrgdv.in' is different from reference."

        str_in.pop(1)
        ref_str.pop(1)

        # Only keep the "outdata/out_POT*\n" from the path for comparison
        for i, _ in enumerate(str_in):
            str_in[i] = "/".join(str_in[i].split("/")[-2:])
            ref_str[i] = "/".join(ref_str[i].split("/")[-2:])

        assert str_in == ref_str, "'mrgdv.in' is different from reference."

    logger.info("Verified inputs successfully")


@pytest.fixture
def mock_anaddb(mocker, abinit_test_dir, abinit_integration_tests):
    """
    This fixture allows one to mock running Anaddb.

    It works by monkeypatching (replacing) calls to run_anaddb.

    The primary idea is that instead of running Anaddb to generate the output files,
    reference files will be copied into the directory instead.
    """
    import atomate2.abinit.files
    import atomate2.abinit.jobs.anaddb
    import atomate2.abinit.run

    # Wrap write_anaddb_input_set to check inputs after writing
    def wrapped_write_anaddb_input_set(*args, **kwargs) -> None:
        from jobflow import CURRENT_JOB

        name = CURRENT_JOB.job.name
        index = CURRENT_JOB.job.index
        ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]

        atomate2.abinit.files.write_anaddb_input_set(*args, **kwargs)
        check_anaddb_inputs(ref_path)

    mocker.patch.object(
        atomate2.abinit.jobs.anaddb,
        "write_anaddb_input_set",
        wrapped_write_anaddb_input_set,
    )

    if not abinit_integration_tests:
        # Mock anaddb run by copying reference output files
        def mock_run_anaddb(
            wall_time: float | None = None, start_time: float | None = None
        ) -> None:
            from jobflow import CURRENT_JOB

            name = CURRENT_JOB.job.name
            index = CURRENT_JOB.job.index
            ref_path = abinit_test_dir / _REF_PATHS[name][str(index)]
            check_anaddb_inputs(ref_path)
            fake_run_abinit(ref_path)

        mocker.patch.object(atomate2.abinit.run, "run_anaddb", mock_run_anaddb)
        mocker.patch.object(atomate2.abinit.jobs.anaddb, "run_anaddb", mock_run_anaddb)

    def _run(ref_paths: dict, fake_run_anaddb_kwargs: dict | None = None) -> None:
        if fake_run_anaddb_kwargs is None:
            fake_run_anaddb_kwargs = {}
        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_ANADDB_KWARGS.update(fake_run_anaddb_kwargs)

    yield _run

    mocker.stopall()
    _REF_PATHS.clear()
    _FAKE_RUN_ANADDB_KWARGS.clear()


def check_anaddb_inputs(
    ref_path: str | Path,
    check_inputs: Sequence[Literal["anaddb.in"]] = _ANADDB_FILES,
) -> None:
    """
    Verify anaddb input files against reference inputs.

    Parameters
    ----------
    ref_path
        Path to reference directory containing input files.
    check_inputs
        Sequence of input file names to check.
    """
    ref_path = Path(ref_path)

    if "anaddb.in" in check_inputs:
        check_anaddb_in(ref_path)

    if "anaddb_input.json" in check_inputs:
        check_anaddb_input_json(ref_path)

    logger.info("Verified inputs successfully")


def convert_file_to_dict(file_path: str) -> dict:
    """
    Convert an anaddb input file to a dictionary.

    Parameters
    ----------
    file_path
        Path to the anaddb input file (can be gzipped).

    Returns
    -------
    dict
        Dictionary representation of the anaddb input file.
    """
    from monty.io import zopen

    result_dict: dict = {}

    if file_path.endswith(".gz"):
        file_opener = zopen
        mode = "rt"  # Read text mode for gzip
    else:
        file_opener = open
        mode = "r"

    with file_opener(file_path, mode) as file:
        current_key = None
        for line in file:
            if "#" in line or len(line) == 1:
                continue
            sl = line.strip().split(" ", 1)
            if is_anaddb_var(sl[0]) and len(sl) > 1:
                try:
                    result_dict[sl[0]] = int(sl[1])
                except ValueError:
                    result_dict[sl[0]] = sl[1]
            elif is_anaddb_var(sl[0]) and len(sl) == 1:
                current_key = sl[0]
                result_dict[current_key] = []
            elif current_key is not None:
                result_dict[current_key].append([float(t) for t in line.split()])
    return result_dict


def check_anaddb_in(ref_path: str | Path) -> None:
    """
    Verify anaddb.in input file against reference.

    Parameters
    ----------
    ref_path
        Path to reference directory containing the reference anaddb.in file.
    """
    user = convert_file_to_dict("anaddb.in")
    ref = convert_file_to_dict(str(ref_path / "inputs" / "anaddb.in.gz"))
    assert user == ref, "'anaddb.in' is different from reference."


def check_anaddb_input_json(ref_path: str | Path) -> None:
    """
    Verify anaddb_input.json against reference.

    Parameters
    ----------
    ref_path
        Path to reference directory containing the reference anaddb_input.json file.
    """
    from abipy.abio.inputs import AnaddbInput
    from monty.serialization import loadfn

    user = loadfn("anaddb_input.json")
    assert isinstance(user, AnaddbInput)
    user_abivars = user.structure.to_abivars()

    ref = loadfn(ref_path / "inputs" / "anaddb_input.json.gz")
    ref_abivars = ref.structure.to_abivars()

    # Check structure
    check_equivalent_frac_coords(user.structure, ref.structure)
    check_equivalent_znucl_typat(
        user_abivars["znucl"],
        ref_abivars["znucl"],
        user_abivars["typat"],
        ref_abivars["typat"],
    )

    for k, user_v in user_abivars.items():
        if k in ["xred", "znucl", "typat"]:
            continue
        assert k in ref_abivars, f"{k = } is not a key of the reference input."
        ref_v = ref_abivars[k]
        if isinstance(user_v, str):
            assert user_v == ref_v, f"{k = }-->{user_v = } versus {ref_v = }"
        else:
            assert np.allclose(user_v, ref_v), f"{k = }-->{user_v = } versus {ref_v = }"

    # Check anaddb input
    user_args = dict(user.as_dict()["anaddb_args"])
    ref_args = dict(ref.as_dict()["anaddb_args"])
    for k, user_v in user_args.items():
        assert k in ref_args, f"{k = } is not a key of the reference input."
        ref_v = ref_args[k]
        if isinstance(user_v, str):
            assert user_v == ref_v, f"{k = }-->{user_v = } versus {ref_v = }"
        elif user_v is None and ref_v is None:
            continue
        else:
            assert np.allclose(user_v, ref_v), f"{k = }-->{user_v = } versus {ref_v = }"


def copy_abinit_outputs(ref_path: str | Path) -> None:
    """
    Copy ABINIT output files from reference directory to current directory.

    Parameters
    ----------
    ref_path
        Path to reference directory containing output files in the outputs subdirectory.
    """
    import shutil

    from monty.shutil import decompress_file

    ref_path = Path(ref_path)
    output_path = ref_path / "outputs"
    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")
            decompress_file(output_file.name)
    for data_dir in ("indata", "outdata", "tmpdata"):
        ref_data_dir = output_path / data_dir
        if not ref_data_dir.exists():
            # means this ref dir was empty and thus removed
            continue
        for file in ref_data_dir.iterdir():
            if file.is_file():
                shutil.copy(file, data_dir)
                decompress_file(str(Path(data_dir, file.name)))
