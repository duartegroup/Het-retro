# Ring dataset

To retrain the models, split your reaction dataset into reactants and products, tokenize (using function from https://github.com/pschwllr/MolecularTransformer) and save here as separate files with one entry per line (following the format of uspto and recent datasets).

In our work the Ring dataset contained ring formation reactions extracted from CJHIF combined with heterocycle formation reactions from Pistachio (https://www.nextmovesoftware.com/pistachio.html).
The full dataset was split into train, validation and test sets with a 80:10:10 ratio using the Fingerprint Splitter from DeepChem (https://deepchem.readthedocs.io/en/latest/api_reference/splitters.html#fingerprintsplitter) based on the reaction product.


