# Basic driver for performing a multiscale many-body expansion based permuted adaptive parititioning (MMBE-PAP) energy calculation

# TODO: Change output file writing so that layers are written in order. Current implementation will write all non-buffer layers first, then all buffer layers.

# Package imports
import numpy as np
import datetime

# Local imports
from molecular_system import MolecularSystem
from run_parameters import RunParameters
from compute_monomers import compute_monomers
from compute_dimers import compute_dimers
from compute_trimers import compute_trimers
from compute_one_body_corrections import compute_one_body_energy
from compute_two_body_corrections import compute_two_body_energy_corrections
from compute_three_body_corrections import compute_three_body_energy_corrections

# Arbitrary-layer energy driver for model calculations
def energy_driver(molecular_system: MolecularSystem = None,
                  run_params: RunParameters = None,
                  output_file: object = None) -> float :

    """
    Main function for performing a multiscale many-body expansion based permuted adaptive partitioning (MMBE-PAP) energy calculation.

    Args:
        molecular_system: MolecularSystem object containing the molecular system information.
        output_file: File object for writing output data.

    Returns:
        float: The total energy or interaction energy of the system.

    Raises:
        ValueError: If an invalid run type is specified.
    """
    
    # Start the timer
    start_time = datetime.datetime.now()

    # Get the total number of fragments (this will be used frequently, so avoid frequent calling)
    total_num_frags = molecular_system.num_frags()

    # Determine if this is an interaction only calculation
    # TODO: Move this to RunParameters
    if "interaction_only" not in run_params["keywords"]:
        return_interaction_only = False
    else:
        return_interaction_only = run_params["keywords"]["interaction_only"]
        # Debugging output to console (TODO: remove in production)
        print(f"Interaction only calculation: {return_interaction_only}")
    
    # Build the arrays for the one, two, and three body energy contributions for each layer
    one_body_layer_energies = np.zeros((run_params.num_layers(), total_num_frags)) # 2D array for one-body energies
    two_body_layer_energies = np.zeros((run_params.num_layers(), total_num_frags, total_num_frags)) # 3D array for dimer energies
    three_body_layer_energies = np.zeros((run_params.num_layers(), total_num_frags, total_num_frags, total_num_frags)) # 4D array for trimer energies
    two_body_layer_energy_corrections = np.zeros((run_params.num_layers(), total_num_frags, total_num_frags)) # 3D array for dimer corrections
    three_body_layer_energy_corrections = np.zeros((run_params.num_layers(), total_num_frags, total_num_frags, total_num_frags)) # 4D array for trimer corrections
    one_body_layer_energy = np.zeros(run_params.num_layers()) # 1D array for one-body energies
    two_body_layer_energy_correction = np.zeros(run_params.num_layers()) # 1D array for two-body corrections
    three_body_layer_energy_correction = np.zeros(run_params.num_layers()) # 1D array for three-body corrections

    # Initialize total energy and interaction energy for the system
    total_energy = 0.0
    interaction_energy = 0.0

    # Compute the monomer energies
    monomer_compute_time_start = datetime.datetime.now()
    compute_monomers(molecular_system, run_params, one_body_layer_energies)
    output_file.write("Monomer energy computation time: " + str(datetime.datetime.now() - monomer_compute_time_start) + "\n")
    
    # Compute the dimer energies
    dimer_compute_time_start = datetime.datetime.now()
    compute_dimers(molecular_system, run_params, two_body_layer_energies, two_body_layer_energy_corrections)
    output_file.write("Dimer energy computation time: " + str(datetime.datetime.now() - dimer_compute_time_start) + "\n")
    
    # Compute trimer energies (if MMBE order is 3)
    if run_params.mmbe_order() == 3:
        trimer_compute_time_start = datetime.datetime.now()
        compute_trimers(molecular_system, run_params, three_body_layer_energies, three_body_layer_energy_corrections)
        output_file.write("Trimer energy computation time: " + str(datetime.datetime.now() - trimer_compute_time_start) + "\n")

    # Compute total one-body energy
    one_body_compute_time_start = datetime.datetime.now()
    one_body_energy = compute_one_body_energy(molecular_system, run_params, one_body_layer_energies)
    total_energy += one_body_energy
    output_file.write("One-body energy computation time: " + str(datetime.datetime.now() - one_body_compute_time_start) + "\n")
    output_file.write("One-body energy: " + str(one_body_energy) + "\n")

    # Compute total two-body correction to the energy
    two_body_compute_time_start = datetime.datetime.now()
    two_body_energy_correction = compute_two_body_energy_corrections(molecular_system, run_params, one_body_layer_energies, two_body_layer_energies, two_body_layer_energy_corrections)
    total_energy += two_body_energy_correction
    interaction_energy += two_body_energy_correction
    output_file.write("Two-body energy computation time: " + str(datetime.datetime.now() - two_body_compute_time_start) + "\n")
    output_file.write("Two-body energy correction: " + str(two_body_energy_correction) + "\n")

    # Compute total three-body correction to the energy (if MMBE order is 3)
    if run_params.mmbe_order() == 3:
        three_body_compute_time_start = datetime.datetime.now()
        three_body_energy_correction = compute_three_body_energy_corrections(molecular_system, run_params, one_body_layer_energies, two_body_layer_energies, three_body_layer_energies, three_body_layer_energy_corrections)
        total_energy += three_body_energy_correction
        interaction_energy += three_body_energy_correction
        output_file.write("Three-body energy computation time: " + str(datetime.datetime.now() - three_body_compute_time_start) + "\n")
        output_file.write("Three-body energy correction: " + str(three_body_energy_correction) + "\n")

    # Write the total energy and runtime to the output file
    output_file.write("Total energy: " + str(total_energy) + "\n")
    output_file.write("Total interaction energy: " + str(interaction_energy) + "\n")
    output_file.write("Total time: " + str(datetime.datetime.now() - start_time) + "\n")

    # Print some final data to the console
    print("One-body energy: ", one_body_energy)
    print("Two-body energy correction: ", two_body_energy_correction)
    if run_params.mmbe_order() == 3:
        print("Three-body energy correction: ", three_body_energy_correction)
    print("Total energy: ", total_energy)
    print("Total interaction energy: ", interaction_energy)
    print("Total time: ", datetime.datetime.now() - start_time)
    print("Run completed successfully.")

    # Return either the total energy or the interaction energy, depending on the flag
    if return_interaction_only:
        return interaction_energy
    else:
        return total_energy