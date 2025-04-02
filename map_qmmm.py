# Entry point for the Multiscale Adaptive Partitioning QM/MM program. 
# Builds the molecular system, run parameters, etc. and then calls the appropriate driver code.

# Import the necessary modules
import os
import sys
import datetime
import numpy as np
import json

# Local imports
from molecular_system import MolecularSystem
from run_parameters import RunParameters
from set_layer_indices import set_layer_indices

# Import the appropriate driver code
from md_driver import md_driver
from single_point_driver import single_point_driver
from optimize import optimization_driver

# Entry point for the Multiscale Adaptive Partitioning QM/MM program
if __name__ == "__main__":

    """ Main entry point for the Multiscale Adaptive Partitioning QM/MM program.

    This program reads in a JSON input file, builds the molecular system and run parameters,
    sets the layer indices for the fragments, and then calls the appropriate driver code.
    
    """

    # Check if the input file is provided as a command line argument
    if len(sys.argv) < 2:
        print("Usage: python map_qmmm.py <input_file.json>")
        sys.exit(1)
        
    # Read the input file
    with open(sys.argv[1], "r") as json_input_file:
        input_data = json.load(json_input_file)

    # Open the output file for printing relevant data
    output_file_path = os.path.splitext(sys.argv[1])[0] + ".out"

    # If the output file path already exists, add a datetime stamp to the filename
    if os.path.exists(output_file_path):
        output_file_path = os.path.splitext(sys.argv[1])[0] + "_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".out"
    output_file = open(output_file_path, "w")

    # Start the timer
    start_time = datetime.datetime.now()

    # Build the molecular system
    molecular_system = MolecularSystem(input_data)

    # Get run parameters (dictionary) from the input file
    run_params = RunParameters(input_data)

    # Set the proper layer indices for the fragments
    set_layer_indices(molecular_system, run_params)
	
    # Add the file name to the output file 
    file_name = os.path.split(output_file_path)
    output_file.write("File Name: " + str(file_name[1]) + "\n")

    ##### Beginning calculation #####
    output_file.write("Date and time: " + str(datetime.datetime.now()) + "\n")
    output_file.write(f"Beginning " + run_params["driver"] + " calculation" + "\n")
    output_file.write("Total number of fragments: " + str(molecular_system.num_frags()) + "\n")
    output_file.write("Distributed across " + str(run_params.num_layers()) + " layers." + "\n")

    # Determine if this is an interaction only calculation, and if it is then note it in the output file
    # TODO: Move this to RunParameters
    if "interaction_only" not in run_params["keywords"]:
        return_interaction_only = False
    else:
        return_interaction_only = run_params["keywords"]["interaction_only"]
        # Debugging output to console (TODO: remove in production)
        print(f"Interaction only calculation: {return_interaction_only}")
        output_file.write("Note: This is an interaction-only calculation.\n")


    # Call the appropriate driver code
    if run_params["driver"] == "md":
        md_driver(molecular_system, run_params, output_file)
    elif run_params["driver"] == "optimize":
        optimized_geometry = optimization_driver(molecular_system, run_params, output_file)
    elif run_params["driver"] == "gradient":
        gradient = single_point_driver(molecular_system, run_params, runtype = "gradient", output_file = output_file)
        print("Gradient: ", gradient)
        output_file.write(f"Gradient: {gradient}\n")
    elif run_params["driver"] == "energy":
        energy = single_point_driver(molecular_system, run_params, runtype = "energy", output_file = output_file)
        print("Total energy: ", energy)
        output_file.write(f"Total energy: {energy}\n")
    else:
        raise ValueError("Invalid run type (keyword: driver) specified in the input file.")

    # Stop the timer
    end_time = datetime.datetime.now()
    output_file.write(f"Total runtime: {end_time - start_time}")

    # Close the output file
    output_file.close()
