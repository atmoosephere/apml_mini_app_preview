# Basic script for performing a multiscale many-body expansion based permuted adaptive parititioning (MMBE-PAP) energy calculation

# TODO: Change output file writing so that layers are written in order. Current implementation will write all non-buffer layers first, then all buffer layers.

# Package imports
import numpy as np
from numpy.typing import NDArray
import datetime

# Local imports
from molecular_system import MolecularSystem
from run_parameters import RunParameters
from compute_monomers import compute_monomer_gradients
from compute_dimers import compute_dimer_gradients
from compute_trimers import compute_trimer_gradients
from compute_one_body_corrections import compute_one_body_gradient
from compute_two_body_corrections import compute_two_body_gradient_corrections
from compute_three_body_corrections import compute_three_body_gradient_corrections

# Gradient driver
def gradient_driver(molecular_system: MolecularSystem = None,
                    run_params: RunParameters = None,
                    output_file: object = None) -> NDArray:
    
    """
    Main function for performing a multiscale many-body expansion based permuted adaptive partitioning (MMBE-PAP) gradient calculation.

    Args:
        molecular_system: MolecularSystem object containing the molecular system information.
        output_file: File object for writing output data.

    Returns:
        The total gradient of the system (inside of an AtomicResult container).

    Raises:
        ValueError: If an invalid run type is specified.

    """

    # Start the timer
    start_time = datetime.datetime.now()

    # Get the total number of fragments (this will be used frequently, so avoid frequent calling)
    total_num_frags = molecular_system.num_frags()

    # Get the total number of atoms in the system
    total_num_atoms = molecular_system.num_atoms()

    # Determine if this is an interaction only calculation
    # TODO: Move this to RunParameters
    if "interaction_only" not in run_params["keywords"]:
        return_interaction_gradient = False
    else:
        return_interaction_gradient = run_params["keywords"]["interaction_only"]
        # Debugging output to console (TODO: remove in production)
        print(f"Interaction only calculation: {return_interaction_gradient}")
    
    # Build the arrays for the one, two, and three body energy contributions for each layer
    one_body_layer_gradients = np.zeros((run_params.num_layers(), total_num_frags, total_num_atoms, 3), dtype=np.float64) # Array of fragment gradients for each layer
    two_body_layer_gradients = np.zeros((run_params.num_layers(), total_num_frags, total_num_frags, total_num_atoms, 3), dtype=np.float64) # Array of dimer gradients for each layer
    three_body_layer_gradients = np.zeros((run_params.num_layers(), total_num_frags, total_num_frags, total_num_frags, total_num_atoms, 3), dtype=np.float64) # Array of trimer gradients for each layer
    two_body_layer_gradient_corrections = np.zeros((run_params.num_layers(), total_num_frags, total_num_frags, total_num_atoms, 3), dtype=np.float64) # Array of dimer gradient corrections for each layer
    three_body_layer_gradient_corrections = np.zeros((run_params.num_layers(), total_num_frags, total_num_frags, total_num_frags, total_num_atoms, 3), dtype=np.float64) # Array of trimer gradient corrections for each layer
    
    # Initialize total gradient for the system
    total_gradient = np.zeros((total_num_atoms, 3), dtype=np.float64) 
    interaction_gradient = np.zeros((total_num_atoms, 3), dtype=np.float64)

    # Compute the monomer gradients
    monomer_compute_time_start = datetime.datetime.now()
    compute_monomer_gradients(molecular_system, run_params, one_body_layer_gradients)
    output_file.write("Monomer gradient computation time: " + str(datetime.datetime.now() - monomer_compute_time_start) + "\n")
    
    # Compute the dimer gradients
    dimer_compute_time_start = datetime.datetime.now()
    compute_dimer_gradients(molecular_system, run_params, two_body_layer_gradients, two_body_layer_gradient_corrections)
    output_file.write("Dimer gradient computation time: " + str(datetime.datetime.now() - dimer_compute_time_start) + "\n")
    
    # Compute trimer gradients (if MMBE order is 3)
    if run_params.mmbe_order() == 3:
        trimer_compute_time_start = datetime.datetime.now()
        compute_trimer_gradients(molecular_system, run_params, three_body_layer_gradients, three_body_layer_gradient_corrections)
        output_file.write("Trimer gradient computation time: " + str(datetime.datetime.now() - trimer_compute_time_start) + "\n")

    # Compute total one-body gradient
    one_body_compute_time_start = datetime.datetime.now()
    one_body_gradient = compute_one_body_gradient(molecular_system, run_params, one_body_layer_gradients)
    total_gradient += one_body_gradient
    output_file.write("One-body gradient computation time: " + str(datetime.datetime.now() - one_body_compute_time_start) + "\n")
    output_file.write("One-body gradient:")
    output_file.write(str(one_body_gradient) + "\n")

    # Compute total two-body correction to the gradient
    two_body_compute_time_start = datetime.datetime.now()
    two_body_gradient_correction = compute_two_body_gradient_corrections(molecular_system, run_params, one_body_layer_gradients, two_body_layer_gradients, two_body_layer_gradient_corrections)
    total_gradient += two_body_gradient_correction
    interaction_gradient += two_body_gradient_correction
    output_file.write("Two-body gradient computation time: " + str(datetime.datetime.now() - two_body_compute_time_start) + "\n")
    output_file.write("Two-body gradient correction:")
    output_file.write(str(two_body_gradient_correction) + "\n")

    # Compute total three-body correction to the gradient (if MMBE order is 3)
    if run_params.mmbe_order() == 3:
        three_body_compute_time_start = datetime.datetime.now()
        three_body_gradient_correction = compute_three_body_gradient_corrections(molecular_system, run_params, one_body_layer_gradients, two_body_layer_gradients, three_body_layer_gradients, three_body_layer_gradient_corrections)
        total_gradient += three_body_gradient_correction
        interaction_gradient += three_body_gradient_correction
        output_file.write("Three-body gradient computation time: " + str(datetime.datetime.now() - three_body_compute_time_start) + "\n")
        output_file.write("Three-body gradient correction:")
        output_file.write(str(three_body_gradient_correction) + "\n")

    # Print the total gradient
    output_file.write("Total gradient: ")
    output_file.write(str(total_gradient) + "\n")
    output_file.write("Interaction gradient: ")
    output_file.write(str(interaction_gradient) + "\n")
    output_file.write("Total time: " + str(datetime.datetime.now() - start_time) + "\n")

    # Print the total gradient to the console
    print("Total gradient: ")
    print("[")
    for atom_index in range(total_num_atoms):
        # Print the atom gradient to the console in array format
        print("[", total_gradient[atom_index][0], ", ", total_gradient[atom_index][1], ", ", total_gradient[atom_index][2], "],")
    print("]")
    print("Interaction gradient: ")
    print("[")
    for atom_index in range(total_num_atoms):
        # Print the atom gradient to the console in array format
        print("[", interaction_gradient[atom_index][0], ", ", interaction_gradient[atom_index][1], ", ", interaction_gradient[atom_index][2], "],")
    print("]")

    # Return the total gradient
    if return_interaction_gradient:
        return interaction_gradient
    else:
        return total_gradient