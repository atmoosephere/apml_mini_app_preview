# Package import(s)
import copy
import qcengine as qcng
import qcelemental as qcel 
import numpy as np
from numpy.typing import NDArray
# TODO: Add as needed

# Local import(s)
# TODO: Add as needed

# Molecule class
class Molecule():

    # Constructor
    def __init__(self,
                 name: str,
                 geometry: [],
                 symbols: [],
                 molecular_charge: int = 0,
                 molecular_multiplicity: int = 1,
                 layer: int = 1):     

        # Set basic molecule variables
        self.name = name  
        self.geometry = geometry
        self.symbols = symbols
        self.molecular_charge = molecular_charge
        self.molecular_multiplicity = molecular_multiplicity
        self.layer = layer

        # Set the default velocities as a num_total_atoms x 3 matrix to 0
        self.velocities = np.zeros((len(self.symbols), 3))

        # Set the coordinates to the provided geometry
        self.coordinates = np.array(self.geometry).reshape(len(self.symbols), 3)
        # Tracks the number of atoms in a molecule 
        self.atom_quantity = int(len(self.geometry)/3)
        
        # Initialize atomic masses using QCElemental
        self.masses = np.array([qcel.periodictable.to_mass(symbol) 
                                    for symbol in self.symbols])
        
        # Need to get connectivity for MM calculations using QCEngine
        self.connectivity = qcel.molutil.guess_connectivity(symbols = self.symbols, geometry = self.geometry)
        for bond_index in range(len(self.connectivity)):
            self.connectivity[bond_index] = tuple([self.connectivity[bond_index][0], self.connectivity[bond_index][1], 1.0])
       
        # NOTE: This method of instantiating a molecule gives unexpected results 
        # calls similar to the simplified line below also result in similar errors 
        # mol = qcel.models.Molecule(**{"symbols": ["He"], "geometry": [0, 0, 0]})
        # Create the QCElemental molecule reprentation

        self.qcel_mol = qcel.models.Molecule(orient =True,
                                             name = self.name,
                                             geometry = self.geometry, 
                                             symbols = self.symbols,
                                             masses = self.masses, 
                                             connectivity = self.connectivity,
                                             molecular_charge = self.molecular_charge,
                                             molecular_multiplicity = self.molecular_multiplicity,
                                             fix_orientation = True,
                                             fix_com = True,
                                            fix_symmetry = "c1")
        
        # QCEngine string input formatting                     
        def qcel_input_formatter(geometry,symbols,charge,multiplicity):
            symbol_index = 0
            c_index = 0 
            input_string = str(charge) + " " +  str(multiplicity) + "\n"
            while c_index < len(geometry): 
                mol_symbol = symbols[symbol_index]
                x_coord = str(geometry[c_index]) 
                y_coord = str(geometry[c_index+1])
                z_coord = str(geometry[c_index+2])
                input_string += mol_symbol + " " + x_coord + "," + y_coord + "," + z_coord + "\n"  
                c_index+= 3 
                symbol_index += 1 
            return input_string

        # Create a string data input for qcengine call 
        self.input_string = qcel_input_formatter(geometry = self.geometry, symbols = self.symbols, charge = self.molecular_charge, multiplicity= self.molecular_multiplicity)

    # Getter function for molecule layer 
    def get_layer(self):
        return self.layer
    
    # Setter function for molecule layer 
    def set_layer(self, layer):
         self.layer = layer 
    
    # Returns molecule name 
    def get_mol_name(self): 
        return self.name
    
    # Get the number of atoms in the molecule
    def num_atoms(self) -> int:
        return self.atom_quantity
    
    # Get the atomic coordinates of the molecule as a (N, 3) array
    def get_coordinates_matrix(self) -> NDArray:
        
        return self.coordinates
    
    # Get the atomic velocities of the molecule as a (N, 3) array
    def get_velocities_matrix(self) -> NDArray:

        return self.velocities
    
    # Set the coordinates of the molecule
    def set_coordinates_matrix(self, new_coords: NDArray) -> None:
        
        # Check if the new coordinates have the correct shape
        if new_coords.shape != (self.atom_quantity, 3):
            raise ValueError("New coordinates must have shape (N, 3).")
        
        # Set the new coordinates in the internal coordinates matrix
        self.coordinates = new_coords

        # Update the geometry in the QCElemental molecule
        self.geometry = new_coords.flatten().tolist()
        self.qcel_mol.copy(update={"geometry": self.geometry})

    # Set the velocities of the molecule
    def set_velocities_matrix(self, new_velocities: NDArray) -> None:

        # Check if the new velocities have the correct shape
        if new_velocities.shape != (self.atom_quantity, 3):
            raise ValueError("New velocities must have shape (N, 3).")
        
        # Set the new velocities in the internal velocities matrix
        self.velocities = new_velocities

    # Update the coordinates matrix after addition of a delta matrix
    def update_coordinates_matrix(self, delta: NDArray) -> None:

        # Update the internal coordinates matrix by adding the delta
        self.coordinates += delta

        # Update the geometry in the QCElemental molecule
        self.geometry = self.coordinates.flatten().tolist()
        self.qcel_mol.copy(update={"geometry": self.geometry})

    # Update the velocities matrix after addition of a delta matrix
    def update_velocities_matrix(self, delta: NDArray) -> None:

        # Update the internal velocities matrix by adding the delta
        self.velocities += delta

    # Get the atomic masses of the molecule as an array
    def get_atomic_masses(self) -> NDArray:
        return self.masses
    
    # Get the center of mass for the molecule
    def center_of_mass(self) -> NDArray :
        
        # Initialize the center of mass and total mass
        com = np.zeros(3)
        total_mass = np.sum(self.masses)
        
        # Calculate the center of mass
        for i in range(self.atom_quantity):
            com[0] += self.geometry[3 * i] * self.masses[i]
            com[1] += self.geometry[3 * i + 1] * self.masses[i]
            com[2] += self.geometry[3 * i + 2] * self.masses[i]

        return (com / total_mass)


    # Get the energy of the molecule at a given level of theory
    def energy(self, method: str = "", basis_set: str = "") -> float:

        # Initialize the molecule energy and the ASE molecule
        mol_energy = 0.0

        # Set up the molecule for ASE based on the method requested
        if method == "scf":
            # Testing the molecule 
            

            # Do SCF calculation with Psi4 by default
            #model = {"method": method, "basis": basis_set, "keywords": {"scf_type": "direct", "print": 5, "freeze_core": False}}
            #input = qcel.models.AtomicInput(molecule = self.qcel_mol, driver = "energy", model = model)
            #result = qcng.compute(input, "psi4")
            #mol_energy = result.return_result
            
            # SCF Calculation using QCEngine from_data method 
            molecule = qcel.models.Molecule.from_data(self.input_string)
            model = {"method": method, "basis": basis_set, "keywords": {"scf_type": "direct", "print": 5, "freeze_core": False}}
            input = qcel.models.AtomicInput(molecule = molecule, driver = "energy",model = model)
            result = qcng.compute(input, "psi4")
            mol_energy = result.return_result

        # Set up the molecule for MOPAC-based semiempirical calculations
        elif method == "se":
            
            # Do semiempirical calculation with MOPAC
            model = {"method": basis_set}
            input = qcel.models.AtomicInput(molecule = self.qcel_mol, driver = "energy", model = model)
            result = qcng.compute(input, "mopac")
            mol_energy = result.return_result
            

        # Set up the molecule for DFTB calculations
        elif method == "dftb":

            # Do DFTB calculation with xtb
            model = {"method": basis_set}
            input = qcel.models.AtomicInput(molecule = self.qcel_mol, driver = "energy", model = model)
            result = qcng.compute(input, "xtb")
            mol_energy = result.return_result
            

        # Set up the molecule for molecular mechanics calculations
        elif method == "mm":

            # Do molecular mechanics calculation
            model = {"method": basis_set}
            input = qcel.models.AtomicInput(molecule = self.qcel_mol, driver = "energy", model = model)
            result = qcng.compute(input, "rdkit")
            mol_energy = result.return_result

        # Compute dispersion correction using DFTD3
        elif method == "dispersion_correction":

            # Do DFTD3 calculation with Psi4
            model = {"method": "b3lyp-d3", "basis": basis_set}
            input = qcel.models.AtomicInput(molecule = self.qcel_mol, driver = "energy", model = model)
            result = qcng.compute(input, "psi4")
            mol_energy = result.return_result

        # Compute df-mp2 energy using Psi4
        elif method == "mp2":

            # Do DF-MP2 calculation with Psi4
            model = {"method": "mp2", "basis": basis_set}
            input = qcel.models.AtomicInput(molecule = self.qcel_mol, driver = "energy", model = model)
            result = qcng.compute(input, "psi4")
            mol_energy = result.return_result
            
        # Unsupported method requested
        else:
            raise Exception("Invalid method requested for molecule energy calculation.")

        return mol_energy
    
    # Get the gradient of the molecule at a given level of theory
    def gradient(self, method: str = "", basis_set: str = "") -> np.array :

        # Initialize the molecule gradient and the ASE molecule
        mol_gradient = np.zeros((self.atom_quantity, 3))

        # Set up the molecule for ASE based on the method requested
        if method == "scf":
            
            # Do SCF gradient calculation with Psi4 by default
            model = {"method": method, "basis": basis_set, "keywords": {"scf_type": "direct"}}
            input = qcel.models.AtomicInput(molecule = self.qcel_mol, driver = "gradient", model = model)
            result = qcng.compute(input, "psi4")
            print("Results of the Psi4 gradient calculation:", result)
            mol_gradient = result.return_result

        # Set up the molecule for MOPAC-based semiempirical calculations
        elif method == "se":

            # Do SE gradient calculation with MOPAC by default
            model = {"method": method, "basis": basis_set}
            input = qcel.models.AtomicInput(molecule = self.qcel_mol, driver = "gradient", model = model)
            result = qcng.compute(input, "mopac")
            print("Results of the MOPAC gradient calculation:", result)
            mol_gradient = result.return_result

        # Set up the molecule for DFTB calculations
        elif method == "dftb":

            # Do DFTB gradient calculation with xtb
            model = {"method": basis_set}
            input = qcel.models.AtomicInput(molecule = self.qcel_mol, driver = "gradient", model = model)
            result = qcng.compute(input, "xtb")
            print("Results of the DFTB gradient calculation:", result)
            mol_gradient = result.return_result

        # Set up the molecule for molecular mechanics calculations
        elif method == "mm":

            # Do molecular mechanics gradient calculation
            model = {"method": basis_set}
            input = qcel.models.AtomicInput(molecule = self.qcel_mol, driver = "gradient", model = model)
            result = qcng.compute(input, "rdkit")
            print("Results of the MM gradient calculation:", result)
            mol_gradient = result.return_result

        # Unsupported method requested
        else:
            raise Exception("Invalid method requested for molecule gradient calculation.")

        return mol_gradient
    
    # Generic computation interface for the molecule
    def compute(self, 
                method: str = "",           # Computational method (SCF, DFT, etc.)
                basis_set: str = "",        # Basis set to use (STO-3G, 6-31G*, etc.)
                driver: str = "",           # Type of calculation (energy, gradient, hessian, etc.)
                program: str = "",          # Computational chemistry program to use (Psi4, Q-Chem, etc.)
                task_config: dict = None,   # Configuration options for the program (memory, threads, etc.)
                keywords: dict = None,
                return_dict: bool = True) -> qcel.models.AtomicResult | dict:

        # Build task_config and keywords if they are empty
        if task_config is None:
            task_config = {}
        if keywords is None:
            keywords = {"freeze_core": False}

        # Build the input model for the QCEngine calculation
        model = {"method": method, "basis": basis_set}
        input_data = qcel.models.AtomicInput(molecule = self.qcel_mol, driver=driver, model = model, keywords = keywords)
        
        # Perform the QCEngine calculation
        result = qcng.compute(input_data=input_data, 
                              program=program, 
                              raise_error=False, 
                              task_config=task_config, 
                              local_options=None,       # local_options is deprecated
                              return_dict=return_dict)
        
        # Check if an error occurred (first check if return is a dictionary)
        if return_dict:
            if result["error"] is not None:
                raise Exception("Error in QCEngine calculation: ", result["error"])
        else:
            if hasattr(result, "error"):
                raise Exception("Error in QCEngine calculation: " + result.error)
        
        # Return the result
        return result
    
    # Add two molecules together to form a new molecule
    def __add__(self, other):

        new_name = self.name + other.name
        new_coords = [*self.geometry, *other.geometry]  # List concatentation (requires Python 3.9+)
        new_symbols = [*self.symbols, *other.symbols]   # List concatentation (requires Python 3.9+)
        new_charge = self.molecular_charge + other.molecular_charge
        new_layer = 0   # TODO: Should this be the lower or higher index for the pair?

        # Spin multiplicity requires special treatment
        if self.molecular_multiplicity == 1 and other.molecular_multiplicity == 1:
            new_spin = 1
        else:
            # TODO: Account for other spin multiplicities
            raise Exception("Spin multiplicities other than 1 are not currently supported.")
         
        return Molecule(new_name, new_coords, new_symbols, new_charge, new_spin, new_layer)
    
    def get_atomic_masses(self) -> np.ndarray:
        """Get atomic masses for all atoms in the molecule.
        
        Returns:
            ndarray: Array of atomic masses in atomic mass units (shape: [num_atoms])
        """
        return self.masses
    
# Unit test(s)
def test_molecule():
    
    # Define the molecule
    mol = Molecule(name = "water",
                   geometry = np.array([0.000,        
                                    0.000,        
                                    0.000,   
                                    0.469,       
                                    -0.370,        
                                    0.755,     
                                    0.467,       
                                    -0.374,       
                                    -0.754]),
                   symbols = np.array(["O", "H", "H"]),
                   molecular_charge = 0,
                   molecular_multiplicity = 1,
                   layer = 0)
    
    # Perform unit test(s)
    assert mol.get_layer() == 0
    assert mol.get_mol_name() == "water"
    assert mol.num_atoms() == 3
    assert np.allclose(mol.get_atomic_masses(), np.array([1.00782503, 15.99491462,  1.00782503]))
    assert np.allclose(mol.center_of_mass(), np.array([0.0, 0.0, -0.05271191]))
    assert np.allclose(mol.get_coordinate_matrix(), np.array([[ 0.0, 0.957, -0.471],
                                                              [ 0.0, 0.0, 0.0],
                                                              [ 0.0, -0.957, -0.471]]))
    
    # Test energy and gradient calculations with different methods
    assert np.isclose(mol.energy(method = "scf", basis_set = "sto-3g"), -73.86598031243479)
    print("Molecular geometry in numpy array format:", mol.get_coordinate_matrix())
    print("Energy using RHF/6-31G*:", mol.energy(method = "scf", basis_set = "6-31G*"))
    print("Energy using PM7:", mol.energy(method = "se", basis_set = "pm7"))
    print("Energy using UFF:", mol.energy(method = "mm", basis_set = "uff"))
    print("Energy using B3LYP-D3", mol.energy(method = "dispersion_correction", basis_set = "cc-pvdz"))
    print("Gradient using RHF/6-31G*", mol.gradient(method = "scf", basis_set = "6-31G*"))
    assert np.allclose(mol.gradient(method = "scf", basis_set = "6-31G*"), np.array([[ 5.60905307e-17, -1.83205576e+00,  9.17096739e-01],
                                                                                     [-1.15687191e-26,  0.00000000e+00, -1.83419348e+00],
                                                                                     [-5.60905307e-17,  1.83205576e+00,  9.17096739e-01]]))

# Testing code for the Molecule class (reconfigure for pytest later on)


# Test molecule addition
def test_molecule_addition():

    # Define the first molecule
    mol1 = Molecule(name = "water_1",
                    geometry = np.array([0.000,        
                                    0.000,        
                                    0.000,   
                                    0.469,       
                                    -0.370,        
                                    0.755,     
                                    0.467,       
                                    -0.374,       
                                    -0.754]),
                    symbols = np.array(["O", "H", "H"]),
                    molecular_charge = 0,
                    molecular_multiplicity = 1,
                    layer = 0)
    print("Energy of monomer 1:", mol1.energy(method = "scf", basis_set = "6-31G*"))
    print("Results from the monomer 1 energy calculation:", mol1.compute(method = "scf", basis_set = "6-31G*", driver = "energy", program = "psi4", return_dict = True))
    
    # Define the second molecule (shifted 3 angstroms along the x-axis)
    mol2 = Molecule(name = "water_2",
                    geometry = np.array([3.000,        
                                    0.000,        
                                    0.000,    
                                    3.087,        
                                    0.591,       
                                    -0.755,
                                    3.089,
                                    0.592,
                                    0.754]),
                    symbols = np.array(["O", "H", "H"]),
                    molecular_charge = 0,
                    molecular_multiplicity = 1,
                    layer = 1)
    print("Energy of monomer 2:", mol2.energy(method = "scf", basis_set = "6-31G*"))
    print("Results from the monomer 2 energy calculation:", mol2.compute(method = "scf", basis_set = "6-31G*", driver = "energy", program = "psi4", return_dict = True))
    
    # Add the two molecules together
    mol3 = mol1 + mol2
    # print("Molecular geometry in numpy array format:", mol3.get_coordinate_matrix())
    # print("Internal molecular geometry representation:", mol3.qcel_mol.geometry)
    # assert np.allclose(mol3.get_coordinate_matrix(), np.array([[ 0. ,  0.957, -0.471],
    #                                                            [ 0. ,  0.   ,  0.   ],
    #                                                            [ 0. , -0.957, -0.471],
    #                                                            [-3. ,  0.957, -0.471],
    #                                                            [-3. ,  0.   ,  0.   ],
    #                                                            [-3. , -0.957, -0.471]]))
    print("Dimer energy at the RHF/6-31G* level:", mol3.energy(method = "scf", basis_set = "6-31G*"))
    print("Results from the dimer energy calculation:", mol3.compute(method = "scf", basis_set = "6-31G*", driver = "energy", program = "psi4", return_dict = True))

    # Calculate the dimer interaction energy
    mol1_energy = mol1.energy(method = "scf", basis_set = "6-31G*")
    mol2_energy = mol2.energy(method = "scf", basis_set = "6-31G*")
    dimer_energy = mol3.energy(method = "scf", basis_set = "6-31G*")
    print("Monomer 1 energy:", mol1_energy)
    print("Monomer 2 energy:", mol2_energy)
    print("Dimer energy:", dimer_energy)
    print("Dimer interaction energy:", dimer_energy - mol1_energy - mol2_energy)
"""
# Test two distant water molecules
def test_two_distant_water_molecules():
    
    # Define the first molecule
    mol1 = Molecule(name = "water",
                    geometry = np.array([0.0, 0.957, -0.471, 0.0, 0.0, 0.0, 0.0, -0.957, -0.471]),
                    symbols = np.array(["H", "O", "H"]),
                    molecular_charge = 0,
                    molecular_multiplicity = 1,
                    layer = 0)
    
    # Define the second molecule (shifted 3 angstroms along the x-axis)
    mol2 = Molecule(name = "water",
                    geometry = np.array([10.0, 0.957, -0.471, 10.0, 0.0, 0.0, 10.0, -0.957, -0.471]),
                    symbols = np.array(["H", "O", "H"]),
                    molecular_charge = 0,
                    molecular_multiplicity = 1,
                    layer = 1)
    
    # Add the two molecules together
    mol3 = mol1 + mol2
    print("Molecular geometry in numpy array format:", mol3.get_coordinate_matrix())
    print("Internal molecular geometry representation:", mol3.qcel_mol.geometry)
    assert np.allclose(mol3.get_coordinate_matrix(), np.array([[ 0. ,  0.957, -0.471],
                                                               [ 0. ,  0.   ,  0.   ],
                                                               [ 0. , -0.957, -0.471],
                                                               [ 10. ,  0.957, -0.471],
                                                               [ 10. ,  0.   ,  0.   ],
                                                               [ 10. , -0.957, -0.471]]))
    print("Dimer energy at the RHF/6-31G* level:", mol3.energy(method = "scf", basis_set = "6-31G*"))

    # Calculate the dimer interaction energy at a distance
    mol1_energy = mol1.energy(method = "scf", basis_set = "6-31G*")
    mol2_energy = mol2.energy(method = "scf", basis_set = "6-31G*")
    dimer_energy = mol3.energy(method = "scf", basis_set = "6-31G*")
    print("Monomer 1 energy:", mol1_energy)
    print("Monomer 2 energy:", mol2_energy)
    print("Dimer energy:", dimer_energy)
    print("Dimer interaction energy:", dimer_energy - mol1_energy - mol2_energy)

# Test limit case of non-interacting dimers when water molecules are 100 angstroms apart
def test_two_non_interacting_water_molecules():
    
    # Define the first molecule
    mol1 = Molecule(name = "water",
                    geometry = np.array([0.0, 0.957, -0.471, 0.0, 0.0, 0.0, 0.0, -0.957, -0.471]),
                    symbols = np.array(["H", "O", "H"]),
                    molecular_charge = 0,
                    molecular_multiplicity = 1,
                    layer = 0)
    
    # Define the second molecule (shifted 3 angstroms along the x-axis)
    mol2 = Molecule(name = "water",
                    geometry = np.array([100.0, 0.957, -0.471, 100.0, 0.0, 0.0, 100.0, -0.957, -0.471]),
                    symbols = np.array(["H", "O", "H"]),
                    molecular_charge = 0,
                    molecular_multiplicity = 1,
                    layer = 1)
    
    # Add the two molecules together
    mol3 = mol1 + mol2
    print("Molecular geometry in numpy array format:", mol3.get_coordinate_matrix())
    print("Internal molecular geometry representation:", mol3.qcel_mol.geometry)
    assert np.allclose(mol3.get_coordinate_matrix(), np.array([[  0. ,   0.957, -0.471],
                                                               [  0. ,   0.   ,  0.   ],
                                                               [  0. , -0.957, -0.471],
                                                               [100. ,   0.957, -0.471],
                                                               [100. ,   0.   ,  0.   ],
                                                               [100. , -0.957, -0.471]]))
    print("Dimer energy at the RHF/6-31G* level:", mol3.energy(method = "scf", basis_set = "6-31G*"))

    # Calculate the dimer interaction energy at a distance
    mol1_energy = mol1.energy(method = "scf", basis_set = "6-31G*")
    mol2_energy = mol2.energy(method = "scf", basis_set = "6-31G*")
    dimer_energy = mol3.energy(method = "scf", basis_set = "6-31G*")
    print("Monomer 1 energy:", mol1_energy)
    print("Monomer 2 energy:", mol2_energy)
    print("Dimer energy:", dimer_energy)
    print("Dimer interaction energy:", dimer_energy - mol1_energy - mol2_energy)

# Test the generic compute interface for the molecule
def test_compute():
    
    # Define the molecule
    mol = Molecule(name = "water",
                   geometry = np.array([0.0, 0.957, -0.471, 0.0, 0.0, 0.0, 0.0, -0.957, -0.471]),
                   symbols = np.array(["H", "O", "H"]),
                   molecular_charge = 0,
                   molecular_multiplicity = 1,
                   layer = 0)
    
    # Perform unit test(s)
    #assert np.isclose(mol.compute(method = "scf", basis_set = "sto-3g", driver = "energy", program = "psi4").return_result, -73.86598031243479)
    #assert np.allclose(mol.compute(method = "scf", basis_set = "6-31G*", driver = "gradient", program = "psi4").return_result, np.array([[ 5.60905307e-17, -1.83205576e+00,  9.17096739e-01],
    #                                                                                                                              [-1.15687191e-26,  0.00000000e+00, -1.83419348e+00],
    #                                                                                                                              [-5.60905307e-17,  1.83205576e+00,  9.17096739e-01]]))

    # Print the results of a RHF 6-31G* energy calculation
    result = mol.compute(method = "scf", basis_set = "6-31G*", driver = "energy", program = "psi4", return_dict = True)

    # Print the energy directly
    print("Energy (RHF/6-31G*):", result["return_result"])

    # Print the results of a RHF STO-3G energy calculation
    result = mol.compute(method = "scf", basis_set = "sto-3g", driver = "energy", program = "psi4", return_dict = True)

    # Print the energy directly
    print("Energy (RHF/STO-3G):", result["return_result"])
""" 
"""
    # Print the results of an AM1 semiempirical energy calculation as a dictionary
    result = mol.compute(method = "am1", basis_set = "", driver = "energy", program = "mopac", return_dict = True)

    for key in result:
        print(key, ":", result[key])

    # Print the energy directly
    print("Energy:", result["return_result"])


    # Print the results of a MP2 gradient calculation as a dictionary
    result = mol.compute(method = "mp2", basis_set = "6-31G*", driver = "gradient", program = "psi4", return_dict = True)

    for key in result:
        print(key, ":", result[key])

    # Print the gradient directly
    # NOTE: Convert the 1D array to a 3x3 matrix for easier viewing
    print("Gradient:", np.array(result["return_result"]).reshape(3, 3))

def test_mol_symm():

    # Define the molecule
    mol = Molecule(name = "water",
                   geometry = np.array([3.774, -2.093, -0.106, 3.55, -2.923, 0.292, 2.957, -1.619, -0.185]),
                   symbols = np.array(["O", "H", "H"]),
                   molecular_charge = 0,
                   molecular_multiplicity = 1,
                   layer = 0)
    
    # Perform unit test (Test energy with forced C1 symmetry keyword for psi4)
    keywords = {"scf_type": "df"}
    task_config = {"memory": 2 * 1024, "ncores": 2}
    results = mol.compute(method = "scf", basis_set = "6-31G", driver = "energy", program = "psi4", task_config = task_config, keywords = keywords, return_dict = True)
    print("Energy with C1 symmetry keyword:", results["return_result"])

    for key in results:
        print(key, ":", results[key])
        
# Test molecule addition to verify which atoms are assigned to which segment
def test_addition():

    # Define the first molecule
    mol1 = Molecule(name = "water",
                    geometry = np.array([0.0, 0.957, -0.471, 0.0, 0.0, 0.0, 0.0, -0.957, -0.471]),
                    symbols = np.array(["H", "O", "H"]),
                    molecular_charge = 0,
                    molecular_multiplicity = 1,
                    layer = 0)
    
    # Define the second molecule (shifted 3 angstroms along the x-axis)
    mol2 = Molecule(name = "water",
                    geometry = np.array([3.0, 0.957, -0.471, 3.0, 0.0, 0.0, 3.0, -0.957, -0.471]),
                    symbols = np.array(["H", "O", "H"]),
                    molecular_charge = 0,
                    molecular_multiplicity = 1,
                    layer = 1)
    
    # Add the two molecules together
    mol3 = mol1 + mol2
    print("Molecular geometry in numpy array format:", mol3.get_coordinate_matrix())
    print("Internal molecular geometry representation:", mol3.qcel_mol.geometry)
    assert np.allclose(mol3.get_coordinate_matrix(), np.array([[ 0. ,  0.957, -0.471],
                                                               [ 0. ,  0.   ,  0.   ],
                                                               [ 0. , -0.957, -0.471],
                                                               [ 3. ,  0.957, -0.471],
                                                               [ 3. ,  0.   ,  0.   ],
                                                               [ 3. , -0.957, -0.471]]))

    # Define the third molecule (shifted 3 angstroms along the x-axis)
    mol4 = Molecule(name = "water",
                    geometry = np.array([6.0, 0.957, -0.471, 6.0, 0.0, 0.0, 6.0, -0.957, -0.471]),
                    symbols = np.array(["H", "O", "H"]),
                    molecular_charge = 0,
                    molecular_multiplicity = 1,
                    layer = 2)
    
    # Test trimer addition
    trimer = mol1 + mol2 + mol4
    print("Molecular geometry in numpy array format:", trimer.get_coordinate_matrix())
    print("Internal molecular geometry representation:", trimer.qcel_mol.geometry)
    assert np.allclose(trimer.get_coordinate_matrix(), np.array([[ 0. ,  0.957, -0.471],
                                                                [ 0. ,  0.   ,  0.   ],
                                                                [ 0. , -0.957, -0.471],
                                                                [ 3. ,  0.957, -0.471],
                                                                [ 3. ,  0.   ,  0.   ],
                                                                [ 3. , -0.957, -0.471],
                                                                [ 6. ,  0.957, -0.471],
                                                                [ 6. ,  0.   ,  0.   ],
                                                                [ 6. , -0.957, -0.471]]))

def test_mol_symm():

    # Define the molecule
    mol = Molecule(name = "water",
                   geometry = np.array([3.774, -2.093, -0.106, 3.55, -2.923, 0.292, 2.957, -1.619, -0.185]),
                   symbols = np.array(["O", "H", "H"]),
                   molecular_charge = 0,
                   molecular_multiplicity = 1,
                   layer = 0)
    
    # Perform unit test (Test energy with forced C1 symmetry keyword for psi4)
    keywords = {"scf_type": "df"}
    task_config = {"memory": 2 * 1024, "ncores": 2}
    results = mol.compute(method = "scf", basis_set = "6-31G", driver = "energy", program = "psi4", task_config = task_config, keywords = keywords, return_dict = True)
    print("Energy with C1 symmetry keyword:", results["return_result"])

    for key in results:
        print(key, ":", results[key])"
"""

# Run the unit test(s) if the module is called directly
if __name__ == '__main__':

    # Run the test(s)
    #test_molecule()
    #test_molecule_addition()
    #test_two_distant_water_molecules()
    #test_two_non_interacting_water_molecules()
    #test_compute()

    # Test symmetry keyword
    #test_mol_symm()

    #test_addition()

    # Test addition
    test_molecule_addition()

    # Print message
    print("All tests passed!")
