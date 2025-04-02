#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 9 18:19:35 2024

@author: julia
"""

# Molecular system class

# Package imports
import numpy as np
from numpy.typing import NDArray

# Local imports
from molecule import Molecule

class MolecularSystem(Molecule): 
    #Class variables 
    system_name = ""
    keywords = dict [str, any]
    driver = ""
    molecules = []
    system_size = int 

    # Constructor
    def __init__(self,
                 input_data, 
                 ):
        # Review the shape of the input data and redo these class assignments as needed 

        self.system_name= input_data['system_name']
        self.keywords = input_data['keywords']
        self.driver = input_data['driver']
        self.fragments = input_data['fragments']
        self.system_size = int(len(input_data['fragments']))
        self.q1_center_index = self.keywords["q1_center_index"]
        self.atom_start_indices = []

        # Construct internal lists of layer indices and radii
        self.layers = [i for i in range(len(self.keywords["layer_keywords"]))]  # Layers are indexed from 0 to num_layers - 1
        self.layer_radii = []

        #Create molecule instances from fragments 
        atom_counter = 0 # Counter for tracking starting atom indices of each molecule
        if self.fragments != []:
                
            # Initialize empty list to hold molecule instances
            molecules_list = []

            # Loop over fragments and create molecule instances 
            for i in range(self.system_size):

                # Set the starting atom index for the current molecule
                self.atom_start_indices.append(atom_counter)
                
                # Get the current fragment
                current_fragment = self.fragments[i]

                # Get the necessary data for the current fragment
                name = current_fragment["name"]
                symbols = current_fragment["symbols"]
                charge = current_fragment["molecular_charge"]
                spin = current_fragment["molecular_multiplicity"]
                atom_coords = current_fragment["geometry"]
                layer = current_fragment["layer"]

                # Create a molecule instance for the fragment
                current_molecule = Molecule(name, atom_coords, symbols, charge, spin, layer)
                
                # Append the current molecule instance to the list of molecules
                molecules_list.append(current_molecule)

                # Increment atom counter by number of atoms in current molecule
                atom_counter += len(symbols) # Number of atoms in molecule == number of symbols

            molecules_list.sort(key=lambda x: x.layer, reverse=False)     
            self.molecules = molecules_list 

            # Add coordinates and velocities for all atoms in the total system
            self.coordinates = np.zeros((atom_counter, 3), dtype=np.float64)
            self.velocities = np.zeros((atom_counter, 3), dtype=np.float64)  # Velocities are initially set to zero

            # Loop over fragments and add atom coordinates to the total coordinates array
            atom_counter = 0
            for i in range(self.system_size):
                current_mol = self.molecules[i]
                num_atoms = current_mol.num_atoms()
                self.coordinates[atom_counter:atom_counter + num_atoms] = current_mol.get_coordinates_matrix()
                atom_counter += num_atoms

            # Build the layer radii list from self.keywords
            for i in range(len(self.layers)):
                self.layer_radii.append(self.keywords["layer_keywords"][i]["radius"])
        
        # If there are no fragments, raise an error
        else:
            raise ValueError("No fragments provided in input data.")

    # Returns a subset of the total fragments in the instance based on given layer bounds 
    def subsystem(self, layer_lower_bound_index: int = 1, layer_upper_bound_index: int = 1) -> list[Molecule] :
        new_system = []
        size = self.system_size
        for i in range(size): 
            current_mol = self.molecules[i]
            current_layer = current_mol.get_layer()
            if current_layer <= layer_upper_bound_index and current_layer >= layer_lower_bound_index:
                  new_system.append(current_mol)
        return new_system
    
    # Returns the center of mass of the molecule designated the Q1 center molecule 
    def q1_center(self) -> NDArray: 
        return self.molecules[self.q1_center_index].center_of_mass()
    
    # Sort molecules array by layer in decreasing order
    # The molecules are sorted by layer in INCREASING order at construction
    # but this functionality is left here in case it is needed later 
    def sort_by_layer(self, sort_increasing: bool = False):
        
        # Sort the molecules by layer in increasing or decreasing order
        if sort_increasing:
            self.molecules.sort(key=lambda x: x.layer, reverse=False)
        else:
            self.molecules.sort(key=lambda x: x.layer, reverse=True) 

        # Need to update internal coordinates and velocities matrices
        atom_counter = 0
        for i in range(self.system_size):
            current_mol = self.molecules[i]
            num_atoms = current_mol.num_atoms()
            self.coordinates[atom_counter:atom_counter + num_atoms] = current_mol.get_coordinates_matrix()
            self.velocities[atom_counter:atom_counter + num_atoms] = current_mol.get_velocities_matrix()
            atom_counter += num_atoms 

    # Returns the index of the first molecule in a given layer 
    def start_frag_index(self, layer_index: int) -> int: 
        size = self.system_size
        for i in range(size): 
            current_mol = self.molecules[i]
            current_layer = current_mol.get_layer()
            if current_layer == layer_index: 
                return i

    # Returns the number of fragments present within a given layer 
    def num_frags(self,layer_index: int = -1) -> int: 
        size = self.system_size
        if layer_index == -1:
            return size
        else:
            num_frags = 0 
            for i in range(size): 
                current_mol = self.molecules[i]
                current_layer = current_mol.get_layer()
                if current_layer == layer_index:
                    num_frags += 1       
            return num_frags
    
    # Returns the total number of atoms in the molecular system
    def num_atoms(self) -> int: 
        size = self.system_size
        total_atoms = 0
        for i in range(size): 
            current_mol = self.molecules[i]
            total_atoms += current_mol.num_atoms()
        return total_atoms
    
    # Returns the total number of atoms in a given fragment
    def num_atoms_frag(self, frag_index: int) -> int: 
        return self.molecules[frag_index].num_atoms()
    
    # Return the starting atom index of a given fragment
    def start_atom_index(self, frag_index: int) -> int: 
        return self.atom_start_indices[frag_index]
    
    def get_atomic_masses(self) -> NDArray:
        """Get atomic masses for all atoms in the system.
        
        Returns:
            ndarray: Array of atomic masses in atomic mass units (shape: [num_atoms])
        """
        masses = []
        for molecule in self.molecules:
            masses.extend(molecule.get_atomic_masses())
        return np.array(masses)
    
    # Returns the atomic masses of a specific fragment
    def get_fragment_masses(self, frag_index: int) -> NDArray:
        """Get atomic masses for a specific fragment.
        
        Args:
            frag_index: Index of the fragment
            
        Returns:
            ndarray: Array of atomic masses for the fragment (shape: [num_atoms_in_fragment])
        """
        return self.molecules[frag_index].get_atomic_masses()
    
    # Returns the coordinate matrix (in num_total_atoms x 3 format) of the molecular system
    def get_coordinates_matrix(self) -> NDArray:
        """Get the coordinate matrix of the molecular system.
        
        Returns:
            ndarray: Coordinate matrix (shape: [num_atoms, 3])
        """
        return self.coordinates
    
    # Returns the velocity matrix (in num_total_atoms x 3 format) of the molecular system
    def get_velocities_matrix(self) -> NDArray:
        """Get the velocity matrix of the molecular system.
        
        Returns:
            ndarray: Velocity matrix (shape: [num_atoms, 3])
        """
        return self.velocities
    
    # Set the coordinate matrix of the molecular system
    def set_coordinates_matrix(self, new_coords: NDArray = None, update_layers: bool = True) -> None:
        """Set the coordinate matrix of the molecular system.
        
        Args:
            new_coords: New coordinate matrix to set (shape: [num_atoms, 3])
            update_layers: Whether to update the layers of the molecules (default: True)
        """

        # Update the internal coordinates matrix
        self.coordinates = new_coords

        # Now update the coordinates of each molecule
        atom_index = 0
        for molecule in self.molecules:
            num_atoms = molecule.num_atoms()
            molecule.set_coordinates_matrix(new_coords[atom_index:atom_index + num_atoms])
            atom_index += num_atoms

        # Now update the layer assignments if needed
        if update_layers:
            self.set_layer_indices()

    # Set the velocity matrix of the molecular system
    def set_velocities_matrix(self, new_velocities: NDArray):
        """Set the velocity matrix of the molecular system.
        
        Args:
            new_velocities: New velocity matrix to set (shape: [num_atoms, 3])
        """

        # Update the internal velocities matrix
        self.velocities = new_velocities

        # Now update the velocities of each molecule
        atom_index = 0
        for molecule in self.molecules:
            num_atoms = molecule.num_atoms()
            molecule.set_velocities_matrix(new_velocities[atom_index:atom_index + num_atoms])
            atom_index += num_atoms

    # Update the coordinates matrix of the molecular system after adding a delta matrix
    def update_coordinates_matrix(self, delta: NDArray = None, update_layers: bool = True) -> None:
        """Update the coordinate matrix of the molecular system by adding a delta matrix.
        
        Args:
            delta: Delta matrix to add to the current coordinates (shape: [num_atoms, 3])
            update_layers: Whether to update the layers of the molecules (default: True)
        """

        # Update the internal coordinates matrix
        self.coordinates += delta

        # Now update the coordinates of each molecule
        atom_index = 0
        for molecule in self.molecules:
            num_atoms = molecule.num_atoms()
            molecule.update_coordinates_matrix(delta[atom_index:atom_index + num_atoms])
            atom_index += num_atoms
        
        # Now update the layer assignments if needed
        if update_layers:
            self.set_layer_indices()

    # Update the velocity matrix of the molecular system after adding a delta matrix
    def update_velocities_matrix(self, delta: NDArray):
        """Update the velocity matrix of the molecular system by adding a delta matrix.
        
        Args:
            delta: Delta matrix to add to the current velocities (shape: [num_atoms, 3])
        """

        # Update the internal velocities matrix
        self.velocities += delta

        # Now update the velocities of each molecule
        atom_index = 0
        for molecule in self.molecules:
            num_atoms = molecule.num_atoms()
            molecule.update_velocities_matrix(delta[atom_index:atom_index + num_atoms])
            atom_index += num_atoms

    # Set layer indices for the molecular system
    def set_layer_indices(self, sort_layers: bool = True) -> None:

        """Set the layer indices for each fragment in the molecular system.
        
        Determine the layer index for each fragment in the molecular system based on
        the fragment's distance from the Q1 center. The layer index determines the
        level of theory used to compute the fragment's energy.

        Args:
            sort_layers: Whether to sort the layers in increasing order after setting indices (default: True)
        """

        # DEBUG printing TODO: Remove
        print("Setting layer indices for fragments...")
        print("Current layer assignments:")
        for frag_index in range(self.num_frags()):
            print("Fragment: ", frag_index, "is in", self[frag_index].get_layer())

        # Get necessary temporary variables
        q1_center = self.q1_center()
        num_frags = self.num_frags()
        num_layers = len(self.layer_radii)

        # Loop over all fragments in the molecular system
        for frag_index in range(num_frags):

            # Escape if the fragment is the Q1 center fragment
            if frag_index == self.q1_center_index:
                continue

            # Otherwise, compute the fragment's layer index
            else:
                # Get the fragment
                frag = self.molecules[frag_index]

                # Compute the fragment's distance from the Q1 center
                fragment_center = frag.center_of_mass()
                radial_distance = np.linalg.norm(fragment_center - q1_center)

                # Loop over the layers, counting up from the Q1 layer to the outermost layer
                for layer_index in range(num_layers):

                    # The first layer that the fragment falls within is the layer that fragment should be assigned to
                    if radial_distance < self.layer_radii[layer_index]:

                        # Set the fragment's layer index
                        frag.set_layer(layer_index)

                        # Since we found the proper layer index, we can break out of the loop
                        break

        # Sort the molecules by layer in increasing or decreasing order
        if sort_layers:
            self.sort_by_layer(sort_increasing=True)

        # DEBUG printing TODO: Remove
        print("After reassigning layer indices:")
        for frag_index in range(self.num_frags()):
            print("Fragment: ", frag_index, "is in", self.molecules[frag_index].get_layer())

    # Overload the addition operator to combine two molecular systems
    def __add__(self, other):
        """Add two molecular systems together.
        
        Combines coordinates, symbols, charges, spins and masses of both systems.
        
        Args:
            other: Another MolecularSystem to combine with this one
            
        Returns:
            MolecularSystem: Combined system
        """
        new_coords = self.atom_coordinates + other.atom_coordinates
        new_symbols = self.atom_symbols + other.atom_symbols
        new_charge = self.charge + other.charge
        new_spin = self.spin + other.spin
        new_layer = 0
        
        # Create new system
        new_system = MolecularSystem.from_atoms(new_coords, new_symbols, new_charge, new_spin, new_layer)
        
        # Masses will be automatically initialized from symbols
        return new_system

    # Overload __getitem__ method to get fragments from the system
    def __getitem__(self, index):
        return self.molecules[index]
    
# Testing code for MolecularSystem class
if __name__ == "__main__":

    # Example input data
    input_data = {
        "system_name": "TestSystem",
        "keywords": {
            "q1_center_index": 0
        },
        "driver": "energy",
        "fragments": [
            {
                "name": "Fragment0",    
                "symbols": ["O", "H", "H"],
                "molecular_charge": 0,
                "molecular_multiplicity": 1,
                "geometry": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                "layer": 0
            },
            {
                "name": "Fragment1",
                "symbols": ["O", "H", "H"],
                "molecular_charge": 0,
                "molecular_multiplicity": 1,
                "geometry": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.5, 0.5, 0.0],
                "layer": 1
            },
            {
                "name": "Fragment2",
                "symbols": ["C", "H", "H", "H", "H"],
                "molecular_charge": 0,
                "molecular_multiplicity": 1,
                "geometry": [1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.0, 1.5, 1.5, 1.5, 1.0, 1.5, 1.5, 1.5],
                "layer": 2
            }
        ]
    }

    # Create a MolecularSystem instance
    molecular_system = MolecularSystem(input_data)
    
    # Print the total number of atoms in the system
    print("Total number of atoms:", molecular_system.num_atoms())
    
    # Print the coordinates matrix
    print("Coordinates matrix:\n", molecular_system.get_coordinates_matrix())

    # Print the velocity matrix
    print("Velocity matrix:\n", molecular_system.get_velocities_matrix())

    # Get the number of fragments
    print("Number of fragments:", molecular_system.num_frags())

    # Print the masses of the first fragment
    print("Masses of the first fragment:", molecular_system.get_fragment_masses(0))

    # Get the center of mass of the Q1 center molecule
    print("Center of mass of Q1 center molecule:", molecular_system.q1_center())
