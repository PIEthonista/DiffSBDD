QuickVina 2.1 (24 Dec, 2017)


## Installation (Done)
======================
wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1


## Permission (TO RUN)
======================
chmod +x qvina2.1


## Usage
========
>>> $ ./analysis/qvina/qvina2.1 --help_advanced

Input:
  --receptor arg                        rigid part of the receptor (PDBQT)
  --flex arg                            flexible side chains, if any (PDBQT)
  --ligand arg                          ligand (PDBQT)

Search space (required):
  --center_x arg                        X coordinate of the center
  --center_y arg                        Y coordinate of the center
  --center_z arg                        Z coordinate of the center
  --size_x arg                          size in the X dimension (Angstroms)
  --size_y arg                          size in the Y dimension (Angstroms)
  --size_z arg                          size in the Z dimension (Angstroms)

Output (optional):
  --out arg                             output models (PDBQT), the default is 
                                        chosen based on the ligand file name
  --log arg                             optionally, write log file

Advanced options (see the manual):
  --score_only                          score only - search space can be 
                                        omitted
  --local_only                          do local search only
  --randomize_only                      randomize input, attempting to avoid 
                                        clashes
  --weight_gauss1 arg (=-0.035579)      gauss_1 weight
  --weight_gauss2 arg (=-0.005156)      gauss_2 weight
  --weight_repulsion arg (=0.84024500000000002)
                                        repulsion weight
  --weight_hydrophobic arg (=-0.035069000000000003)
                                        hydrophobic weight
  --weight_hydrogen arg (=-0.58743900000000004)
                                        Hydrogen bond weight
  --weight_rot arg (=0.058459999999999998)
                                        N_rot weight

Misc (optional):
  --cpu arg                             the number of CPUs to use (the default 
                                        is to try to detect the number of CPUs 
                                        or, failing that, use 1)
  --seed arg                            explicit random seed
  --exhaustiveness arg (=8)             exhaustiveness of the global search 
                                        (roughly proportional to time): 1+
  --num_modes arg (=9)                  maximum number of binding modes to 
                                        generate
  --energy_range arg (=3)               maximum energy difference between the 
                                        best binding mode and the worst one 
                                        displayed (kcal/mol)

Configuration file (optional):
  --config arg                          the above options can be put here

Information (optional):
  --help                                display usage summary
  --help_advanced                       display usage summary with advanced 
                                        options
  --version                             display program version
