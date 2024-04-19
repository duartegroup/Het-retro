# Ring dataset
# USPTO data
The `cjhif_ring_formations.csv` file contains ring formation reactions extracted from the CJHIF dataset (https://github.com/jshmjs45/data_for_chem/tree/master).

Each row corresponds to a separate reaction and includes:
* Id
* mapped_rxn - atom-mapped reaction (using rxnmapper: https://github.com/rxn4chemistry/rxnmapper)
* confidence - confidence of the atom-mapping
* Rxn - canonicalised reaction
* Reactants (excluding reagents)
* Product

In our work the CJHIF-based dataset was combined with heterocycle formation reactions from Pistachio (https://www.nextmovesoftware.com/pistachio.html).
To reproduce the results, combine `cjhif_ring_formations.csv` with canonicalised Pistachio reactions belonging to the "Heterocycle formations" superclass (class 4) and drop duplicates.
The full dataset then needs to be split into train, validation and test sets with a 80:10:10 ration. In our case this was done using the Fingerprint Splitter from DeepChem (https://deepchem.readthedocs.io/en/latest/api_reference/splitters.html#fingerprintsplitter) based on the reaction product.
The reactions then need to be split into reactants and products, tokenized (using function from https://github.com/pschwllr/MolecularTransformer) and saved as separate files with one entry per line (following the format of uspto and recent datasets).

To train the models using just the CJHIF dataset, follow the approach described above but starting with splitting the `cjhif_ring_formations.csv` dataset into train, validation and test sets. 
