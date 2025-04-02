# Basic script for performing a multiscale many-body expansion based permuted adaptive parititioning (MMBE-PAP) energy calculation

# Gradient driver 
# As this work is yet to be published, this code is not shown 
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

    # Return the total gradient
    if return_interaction_gradient:
        return interaction_gradient
    else:
        return total_gradient
