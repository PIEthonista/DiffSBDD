import os
import re
import tempfile
import numpy as np
import torch
from pathlib import Path
import argparse
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

try:
    import utils
except ModuleNotFoundError as e:
    print(e)


from Bio.PDB import PDBParser
def get_pocket_center(pdb_file):
    """Calculate the geometric center of a pocket from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pdb_file)
    coordinates = []

    # Extract atomic coordinates
    for atom in structure.get_atoms():
        coordinates.append(atom.coord)

    # Calculate the geometric center
    coordinates = np.array(coordinates)
    center = np.mean(coordinates, axis=0)
    return center

def center_ligand(ligand):
    """Center the ligand coordinates around (0, 0, 0)."""
    conf = ligand.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(ligand.GetNumAtoms())])

    # Calculate ligand's center
    ligand_center = np.mean(coords, axis=0)
    print(f">>> Ligand center beforehand: {ligand_center}")

    # Translate coordinates to center the ligand at (0, 0, 0)
    for i in range(ligand.GetNumAtoms()):
        conf.SetAtomPosition(i, coords[i] - ligand_center)
    
    return ligand, ligand_center

def translate_ligand_to_pocket_center(ligand, pocket_center):
    """Translate the ligand to align its center with the pocket center and display the new center."""
    # Get the conformer (3D coordinates) of the ligand
    conf = ligand.GetConformer()
    
    # Iterate over each atom in the ligand
    for i in range(ligand.GetNumAtoms()):
        # Get the current position of the atom
        current_pos = np.array(conf.GetAtomPosition(i))
        
        # Translate the atom position by adding the pocket center (move the ligand)
        conf.SetAtomPosition(i, current_pos + pocket_center)

    # Calculate the new center of the ligand after translation
    new_center = np.mean([np.array(conf.GetAtomPosition(i)) for i in range(ligand.GetNumAtoms())], axis=0)
    
    # Display the new center (xyz coordinates)
    print(f">>> Ligand center after: {new_center}")
    
    # Return the modified ligand with updated positions
    return ligand

def process_sdf_file(sdf_file, pocket_center):
    """Read an SDF file, apply pocket center translation to each molecule, and overwrite the file."""
    supplier = Chem.SDMolSupplier(sdf_file)
    writer = Chem.SDWriter(sdf_file)

    # Loop through all molecules in the SDF file
    for mol in supplier:
        if mol is not None:
            # Step 1: Center ligand around (0, 0, 0)
            mol, ligand_original_center = center_ligand(mol)

            # Step 2: Translate ligand to the pocket center
            mol = translate_ligand_to_pocket_center(mol, pocket_center)

            # Step 3: Write the modified molecule back to the SDF file
            writer.write(mol)

    # Close the writer
    writer.close()






def calculate_smina_score(pdb_file, sdf_file):
    # add '-o <name>_smina.sdf' if you want to see the output
    out = os.popen(f'smina.static -l {sdf_file} -r {pdb_file} '
                   f'--score_only').read()
    matches = re.findall(
        r"Affinity:[ ]+([+-]?[0-9]*[.]?[0-9]+)[ ]+\(kcal/mol\)", out)
    return [float(x) for x in matches]


def smina_score(rdmols, receptor_file):
    """
    Calculate smina score
    :param rdmols: List of RDKit molecules
    :param receptor_file: Receptor pdb/pdbqt file or list of receptor files
    :return: Smina score for each input molecule (list)
    """

    if isinstance(receptor_file, list):
        scores = []
        for mol, rec_file in zip(rdmols, receptor_file):
            with tempfile.NamedTemporaryFile(suffix='.sdf') as tmp:
                tmp_file = tmp.name
                utils.write_sdf_file(tmp_file, [mol])
                scores.extend(calculate_smina_score(rec_file, tmp_file))

    # Use same receptor file for all molecules
    else:
        with tempfile.NamedTemporaryFile(suffix='.sdf') as tmp:
            tmp_file = tmp.name
            utils.write_sdf_file(tmp_file, rdmols)
            scores = calculate_smina_score(receptor_file, tmp_file)

    return scores


def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'obabel {sdf_file} -O {pdbqt_outfile} '
             f'-f {mol_id + 1} -l {mol_id + 1}').read()
    return pdbqt_outfile


def calculate_qvina2_score(receptor_file, sdf_file, out_dir, size=20,
                           exhaustiveness=16, return_rdmol=False):

    receptor_file = Path(receptor_file)
    sdf_file = Path(sdf_file)


# custom testing code
# ===================================
    # Get pocket center
    pocket_center = get_pocket_center(receptor_file)
    print(f"=========================================")
    print(f">>> Pocket center: {pocket_center}")

    # Process the SDF file
    process_sdf_file(sdf_file, pocket_center)
# ===================================



    if receptor_file.suffix == '.pdb':
        # prepare receptor, requires Python 2.7
        receptor_pdbqt_file = Path(out_dir, receptor_file.stem + '.pdbqt')
        os.popen(f'prepare_receptor4.py -r {receptor_file} -O {receptor_pdbqt_file}')
    else:
        receptor_pdbqt_file = receptor_file

    scores = []
    rdmols = []  # for if return rdmols
    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    for i, mol in enumerate(suppl):  # sdf file may contain several ligands
        ligand_name = f'{sdf_file.stem}_{i}'
        # prepare ligand
        ligand_pdbqt_file = Path(out_dir, ligand_name + '.pdbqt')
        out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')

        if out_sdf_file.exists():
            with open(out_sdf_file, 'r') as f:
                scores.append(
                    min([float(x.split()[2]) for x in f.readlines()
                         if x.startswith(' VINA RESULT:')])
                )

        else:
            sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)

            # center box at ligand's center of mass
            cx, cy, cz = mol.GetConformer().GetPositions().mean(0)

            # run QuickVina 2
            out = os.popen(
                f'./qvina/qvina2.1 --receptor {receptor_pdbqt_file} '
                f'--ligand {ligand_pdbqt_file} '
                f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
                f'--size_x {size} --size_y {size} --size_z {size} '
                f'--exhaustiveness {exhaustiveness}'
            ).read()

            # clean up
            ligand_pdbqt_file.unlink()

            if '-----+------------+----------+----------' not in out:
                scores.append(np.nan)
                continue

            out_split = out.splitlines()
            best_idx = out_split.index('-----+------------+----------+----------') + 1
            best_line = out_split[best_idx].split()
            assert best_line[0] == '1'
            scores.append(float(best_line[1]))

            out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
            if out_pdbqt_file.exists():
                os.popen(f'obabel {out_pdbqt_file} -O {out_sdf_file}').read()

                # clean up
                out_pdbqt_file.unlink()

        if return_rdmol:
            rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
            rdmols.append(rdmol)

    if return_rdmol:
        return scores, rdmols
    else:
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser('QuickVina evaluation')
    parser.add_argument('--pdbqt_dir', type=Path,
                        help='Receptor files in pdbqt format')
    parser.add_argument('--sdf_dir', type=Path, default=None,
                        help='Ligand files in sdf format')
    parser.add_argument('--sdf_files', type=Path, nargs='+', default=None)
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--write_csv', action='store_true')
    parser.add_argument('--write_dict', action='store_true')
    parser.add_argument('--dataset', type=str, default='moad')
    args = parser.parse_args()

    assert (args.sdf_dir is not None) ^ (args.sdf_files is not None)

    args.out_dir.mkdir(exist_ok=True)

    results = {'receptor': [], 'ligand': [], 'scores': []}
    results_dict = {}
    sdf_files = list(args.sdf_dir.glob('[!.]*.sdf')) \
        if args.sdf_dir is not None else args.sdf_files
    pbar = tqdm(sdf_files)
    for sdf_file in pbar:
        pbar.set_description(f'Processing {sdf_file.name}')

        if args.dataset == 'moad':
            """
            Ligand file names should be of the following form:
            <receptor-name>_<pocket-id>_<some-suffix>.sdf
            where <receptor-name> and <pocket-id> cannot contain any 
            underscores, e.g.: 1abc-bio1_pocket0_gen.sdf
            """
            ligand_name = sdf_file.stem
            receptor_name, pocket_id, *suffix = ligand_name.split('_')
            suffix = '_'.join(suffix)
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
        elif args.dataset == 'crossdocked':
            ligand_name = sdf_file.stem
            receptor_name = ligand_name[:-4]
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')

        # try:
        # calculate [qvina scores] for all ligands generated for 1 pocket
        scores, rdmols = calculate_qvina2_score(
            receptor_file, sdf_file, args.out_dir, return_rdmol=True)
        # except AttributeError as e:
        #     print(e)
        #     continue
        results['receptor'].append(str(receptor_file))
        results['ligand'].append(str(sdf_file))
        results['scores'].append(scores)

        if args.write_dict:
            results_dict[ligand_name] = {
                'receptor': str(receptor_file),
                'ligand': str(sdf_file),
                'scores': scores,
                'rmdols': rdmols
            }

    if args.write_csv:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(Path(args.out_dir, 'qvina2_scores.csv'))

    if args.write_dict:
        torch.save(results_dict, Path(args.out_dir, 'qvina2_scores.pt'))
