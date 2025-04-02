# Basic driver for performing a multiscale many-body expansion based permuted adaptive parititioning (MMBE-PAP) energy calculation
# Arbitrary-layer energy driver for model calculations
# As this work is yet to the published, the code for this function is not shown 

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
  
    # Return either the total energy or the interaction energy, depending on the flag
    if return_interaction_only:
        return interaction_energy
    else:
        return total_energy
