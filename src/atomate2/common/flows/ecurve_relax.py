from atomate2.vasp.jobs.core import StaticMaker
from jobflow import Flow, Maker, Response, job
from pymatgen.core.structure import Structure
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings("ignore")
import numpy as np

def deform_structure(structure, type, by, test=True):
    if test:
        # Compress or extend the parameter of the conventional cell
        if type == 'compression':
            type_sign = -1
        elif type == 'tension':
            type_sign = 1
        else:
            raise ValueError(f"type should be either compression or tension instead of {type =}")
        
        a_new = structure.lattice.a + (by*type_sign)

        # Find the new lattice with the same angle and orientation as the initial one
        matrix_div = np.ones((3,3))
        matrix_div[0,:] = 1/structure.lattice.parameters[0]
        matrix_div[1,:] = 1/structure.lattice.parameters[1]
        matrix_div[2,:] = 1/structure.lattice.parameters[2]
        matrix_mul = np.ones((3,3))
        matrix_mul[0,:] = a_new
        matrix_mul[1,:] = a_new
        matrix_mul[2,:] = a_new
        matrix_new = structure.lattice.matrix*matrix_div*matrix_mul
    
        return Structure(
            lattice = matrix_new,
            species = structure.species,
            coords  = structure.frac_coords,
            to_unit_cell = False,
            coords_are_cartesian = False
        )

    # Find the parameters of Antoine's cell and of the conventional cell
    b1 = np.min(structure.lattice.parameters[:3])
    b1_arg = np.argmin(structure.lattice.parameters[:3])
    c = np.max(structure.lattice.parameters[:3])
    c_arg = np.argmax(structure.lattice.parameters[:3])
    for param, param_arg in zip(structure.lattice.parameters[:3], [0,1,2]):
        if param != b1 and param != c:
            b2 = param
            b2_arg = param_arg
    a = np.mean([b1/np.sqrt(3/2), b2/2*np.sqrt(2)])

    # Compress or extend the parameter of the conventional cell
    if type == 'compression':
        type_sign = -1
    elif type == 'tension':
        type_sign = 1
    else:
        raise ValueError(f"type should be either compression or tension instead of {type =}")
    a_new = a + (by*type_sign)
    b1_new = a_new*np.sqrt(3/2)
    b2_new = a_new*2/np.sqrt(2)

    parameters_new = np.zeros(3)
    parameters_new[b1_arg] = b1_new
    parameters_new[b2_arg] = b2_new
    parameters_new[c_arg] = c

    # Find the new lattice with the same angle and orientation as the initial one
    matrix_div = np.ones((3,3))
    matrix_div[0,:] = 1/structure.lattice.parameters[0]
    matrix_div[1,:] = 1/structure.lattice.parameters[1]
    matrix_div[2,:] = 1/structure.lattice.parameters[2]
    matrix_mul = np.ones((3,3))
    matrix_mul[0,:] = parameters_new[0]
    matrix_mul[1,:] = parameters_new[1]
    matrix_mul[2,:] = parameters_new[2]
    matrix_new = structure.lattice.matrix*matrix_div*matrix_mul
    
    return Structure(
        lattice = matrix_new,
        species = structure.species,
        coords  = structure.frac_coords,
        to_unit_cell = False,
        coords_are_cartesian = False
    )

def adapt_structure_sf(structure_sf, structure, test=True):
    if test:
        # Find the new sf lattice with the same angle and orientation as the initial one
        matrix_div = np.ones((3,3))
        matrix_div[0,:] = 1/structure_sf.lattice.parameters[0]
        matrix_div[1,:] = 1/structure_sf.lattice.parameters[1]
        matrix_div[2,:] = 1/structure_sf.lattice.parameters[2]
        matrix_mul = np.ones((3,3))
        matrix_mul[0,:] = structure.lattice.a
        matrix_mul[1,:] = structure.lattice.b
        matrix_mul[2,:] = structure.lattice.c
        matrix_new = structure_sf.lattice.matrix*matrix_div*matrix_mul

        return Structure(
            lattice = matrix_new,
            species = structure_sf.species,
            coords  = structure_sf.frac_coords,
            to_unit_cell = False,
            coords_are_cartesian = False
        )


    # Find the new sf lattice with the same angle and orientation as the initial one
    matrix_div = np.ones((3,3))
    matrix_div[0,:] = 1/structure_sf.lattice.parameters[0]
    matrix_div[1,:] = 1/structure_sf.lattice.parameters[1]
    matrix_div[2,:] = 1/structure_sf.lattice.parameters[2]
    matrix_mul = np.ones((3,3))
    matrix_mul[0,:] = structure.lattice.a
    matrix_mul[1,:] = structure.lattice.b
    matrix_mul[2,:] = structure_sf.lattice.c
    matrix_new = structure_sf.lattice.matrix*matrix_div*matrix_mul

    return Structure(
        lattice = matrix_new,
        species = structure_sf.species,
        coords  = structure_sf.frac_coords,
        to_unit_cell = False,
        coords_are_cartesian = False
    )

@job
def check_order(left_output, original_output, right_output, 
                deform_by, static_maker, structure_sf, count=1, last_step=False,
                light_worker_name = None,
                heavy_worker_name = None,
                test = False
                ):
    positions   = np.array(['left', 'original', 'right'])
    energies    = np.array([left_output['energy'], original_output['energy'], right_output['energy']])
    structures  = [left_output['structure'], original_output['structure'], right_output['structure']]
    type_deform = np.array(['compression', 'nothing', 'tension'])

    argsorted = np.argsort(energies)
    energies_sorted = energies[argsorted][::-1] # descending order
    positions_sorted = positions[argsorted][::-1]
    structures_tmp = []
    for arg in argsorted:
        structures_tmp.append(structures[arg])
    structures_sorted = structures_tmp[::-1]
    type_deform_sorted = type_deform[argsorted][::-1]

    if positions_sorted[0] == 'original':
        raise Exception(f"Problem with the order of the energies: {energies_sorted =} -- {positions_sorted =}")
    
    if last_step:
        structure_sf_adapted = adapt_structure_sf(structure_sf, structures_sorted[-1], test = test)
        sf_adapted_static_job = static_maker.make(structure_sf_adapted)
        sf_adapted_static_job.name = "sf adapted structure static job"
        if light_worker_name and heavy_worker_name:
            sf_adapted_static_job.update_config({"manager_config": {"_fworker": heavy_worker_name}}, name_filter="static job", dynamic=False)
        return Response(addition=sf_adapted_static_job)

    if positions_sorted[-1] == 'original':
        structure_new_left = deform_structure(structure=structures_sorted[-1], type='compression', by=deform_by/2, test=test)
        static_job_new_left = static_maker.make(structure=structure_new_left)
        static_job_new_left.name = f"left {count} structure static job"

        structure_new_right = deform_structure(structure=structures_sorted[-1], type='tension', by=deform_by/2, test=test)
        static_job_new_right = static_maker.make(structure=structure_new_right)
        static_job_new_right.name = f"right {count} structure static job"

        flow = Flow([static_job_new_left, static_job_new_right, check_order(
            {'energy': static_job_new_left.output.output.energy, 'structure': static_job_new_left.output.structure},
            {'energy': energies_sorted[-1], 'structure': structures_sorted[-1]},
            {'energy': static_job_new_right.output.output.energy, 'structure': static_job_new_right.output.structure},
            deform_by       = deform_by,
            static_maker    = static_maker,
            structure_sf    = structure_sf,
            count = count+1,
            last_step = True,
            light_worker_name = light_worker_name,
            heavy_worker_name = heavy_worker_name
            )])
        if light_worker_name and heavy_worker_name:
            flow.update_config({"manager_config": {"_fworker": light_worker_name}}, name_filter="check_order", dynamic=False)
            flow.update_config({"manager_config": {"_fworker": heavy_worker_name}}, name_filter="static job", dynamic=False)

        return Response(addition=flow) 

    structure_new = deform_structure(structure=structures_sorted[-1], type=type_deform_sorted[-1], by=deform_by, test=test)
    static_job_new = static_maker.make(structure=structure_new)
    static_job_new.name = f"{positions_sorted[-1]} {count} structure static job"
    jobs = [static_job_new]

    if positions_sorted[-1] == 'right':
        jobs.append(check_order(
            {'energy': energies_sorted[-2], 'structure': structures_sorted[-2]},
            {'energy': energies_sorted[-1], 'structure': structures_sorted[-1]},
            {'energy': static_job_new.output.output.energy, 'structure': static_job_new.output.structure},
            deform_by       = deform_by,
            static_maker    = static_maker,
            structure_sf    = structure_sf,
            count = count+1,
            light_worker_name = light_worker_name,
            heavy_worker_name = heavy_worker_name
            ))
    elif positions_sorted[-1] == 'left':
        jobs.append(check_order(
            {'energy': static_job_new.output.output.energy, 'structure': static_job_new.output.structure},
            {'energy': energies_sorted[-1], 'structure': structures_sorted[-1]},
            {'energy': energies_sorted[-2], 'structure': structures_sorted[-2]},
            deform_by       = deform_by,
            static_maker    = static_maker,
            structure_sf    = structure_sf,
            count = count+1,
            light_worker_name = light_worker_name,
            heavy_worker_name = heavy_worker_name
            ))
    
    flow = Flow(jobs)
    if light_worker_name and heavy_worker_name:
        flow.update_config({"manager_config": {"_fworker": light_worker_name}}, name_filter="check_order", dynamic=False)
        flow.update_config({"manager_config": {"_fworker": heavy_worker_name}}, name_filter="static job", dynamic=False)
    return Response(addition=flow)

@dataclass
class ECurveRelaxMaker(Maker):
    name = "Energy Curve Relax Maker"
    static_maker: StaticMaker = field(
        default_factory=lambda: StaticMaker()
        )
    deform_by = 0.01
    light_worker_name: str | None = field(default=None)
    heavy_worker_name: str | None = field(default=None)

    def make(
            self,
            structure,
            structure_sf,
            test=False
    ):
        # Original SF static
        original_sf_static_job = self.static_maker.make(structure=structure_sf)
        original_sf_static_job.name = "sf structure static job"

        # Original static
        original_static_job = self.static_maker.make(structure=structure)
        original_static_job.name = "original structure static job"

        # Left static (compression)
        left_structure = deform_structure(structure=structure, type='compression', by=self.deform_by)
        left_static_job = self.static_maker.make(structure=left_structure)
        left_static_job.name = "left 1 structure static job"

        # right static (tension)
        right_structure = deform_structure(structure=structure, type='tension', by=self.deform_by)
        right_static_job = self.static_maker.make(structure=right_structure)
        right_static_job.name = "right 1 structure static job"

        # check order of energies
        check_job = check_order(
            left_output     = {'energy': left_static_job.output.output.energy, 'structure': left_static_job.output.structure},
            original_output = {'energy': original_static_job.output.output.energy, 'structure': original_static_job.output.structure},
            right_output    = {'energy': right_static_job.output.output.energy, 'structure': right_static_job.output.structure},
            deform_by       = self.deform_by,
            static_maker    = self.static_maker,
            structure_sf    = structure_sf,
            count = 2,
            light_worker_name = self.light_worker_name,
            heavy_worker_name = self.heavy_worker_name,
            test = test
            )

        if self.light_worker_name and self.heavy_worker_name:
            check_job.update_config({"manager_config": {"_fworker": self.light_worker_name}}, name_filter="check_order", dynamic=False)
            original_sf_static_job.update_config({"manager_config": {"_fworker": self.heavy_worker_name}}, name_filter="static job", dynamic=False)
            original_static_job.update_config({"manager_config": {"_fworker": self.heavy_worker_name}}, name_filter="static job", dynamic=False)
            left_static_job.update_config({"manager_config": {"_fworker": self.heavy_worker_name}}, name_filter="static job", dynamic=False)
            right_static_job.update_config({"manager_config": {"_fworker": self.heavy_worker_name}}, name_filter="static job", dynamic=False)


        jobs = [original_sf_static_job, original_static_job, left_static_job, right_static_job, check_job]


        return Flow(jobs=jobs, output=check_job.output, name=self.name)