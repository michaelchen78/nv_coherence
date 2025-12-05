import numpy as np
import pycce as pc
from ase.build import bulk
from pathlib import Path

# Bath spin types
#              name    spin    gyro       quadrupole (for s>1/2)
# ZFS parameters of NV center in diamond
SPIN_TYPES = [('14N',  1,      1.9338,    20.44),
              ('13C',  1 / 2,  6.72828         ),
              ('e'  ,  1 / 2,  -17608.5962784  )
             ]
D = 2.88 * 1e6 # kHz
E = 0 # kHz
UNIVERSAL_SEED = 8805


# +++++++++++++++++++++++ Auxiliary functions +++++++++++++++++++++++ #
def p1_hyperfine(atoms, on):
    """
    Takes a BathCell with colocated P1 nuclei and electrons and returns either the hyperfine tensor or the zero matrix
    as a Pycce InteractionMap.
    :param atoms: the BathCell
    :param on: if True, returns the hyperfine tensor; otherwise returns zero matrix
    :return: an InteractionMap
    """
    e_idx = np.where(atoms.N == 'e')[0]  # get indices
    p1_idx = np.where(atoms.N == '14N')[0]
    assert np.array_equal(atoms[e_idx].xyz, atoms[p1_idx].xyz ) # ensure nuclei and electrons are colocated
    pairs = list(zip(e_idx, p1_idx))  # pair them up

    if on:
        # hyperfine tensor, from pg 17780 of https://pubs.acs.org/doi/pdf/10.1021/acs.jpcc.2c06145?ref=article_openPDF
        e_p1_interaction = [
            [82.0 * 1e3, 0, 0],
            [0, 82.0 * 1e3, 0],
            [0, 0, 114.0 * 1e3]
        ]
    else:  # in this case, the P1 HF is not considered
        e_p1_interaction = [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]

    imap = pc.InteractionMap()
    for i, j in pairs:
        imap[i, j] = e_p1_interaction  # J_ij between bath spins i and j
    return imap


def append_many(atoms, ids, label):
    """
    Add a spin into a BathCell for every existing spin with index in some list of indices. The new spins are at the same
    locations as the existing spins.
    :param atoms: the BathCell
    :param ids: the list of indices of existing spins
    :param label: the label of the new spin; e.g., "e"
    :return: a new BathCell with the added spins
    """
    ids = np.asarray(ids, dtype=int)
    add = np.zeros(len(ids), dtype=atoms.dtype)
    add['N'] = label
    add['xyz'] = atoms['xyz'][ids]
    return atoms.__class__(array=np.concatenate([atoms.view(np.ndarray),
                                                 add.view(np.ndarray)]))


def get_pulses(pulse_id):
    """
    Given some string pulse id, return the pulse sequence or number of pulses to put into Simulator.compute. See yaml
    for details
    :param pulse_id: the string id
    :return: the pulse sequence or number of pulses
    """
    if pulse_id == "1":
        return 1
    elif pulse_id == "hahn":
        return [pc.Pulse('x', np.pi)]
    else:
        raise Exception("pulse_id not recognized")


# +++++++++++++++++++++++ Primary functions +++++++++++++++++++++++ #
def get_model(model_params):
    """
    Builds a supercell with an NV center and a bath of P1 nuclei and electrons. The supercell is initially generated
    with 14C's, which utilizes PyCCE's native isotopic substitution protocols. These 14Cs are then replaced with P1
    nuclei and electrons. The P1 nuclei and electrons are colocated (identical locations).
    :param model_params: The supercell parameters. This must contain at least p1_conc and size.
    :return: A PyCCE BathCell which is the supercell.
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    # required
    p1_conc = model_params['p1_conc']
    size = model_params['size']
    # optional / defaults
    zdir = model_params.get("zdir", [1, 1, 1])
    seed = model_params.get("seed", UNIVERSAL_SEED)
    c13_conc = model_params.get("c13_conc", 0.011)
    verbose = model_params.get("verbose", False)
    run_id = model_params["run_id"]
    base_out_dir = model_params["base_out_dir"]

    # ================= MAIN CODE ================= #
    # build unit cell
    diamond = pc.read_ase(bulk('C', 'diamond', cubic=True))
    diamond.add_isotopes(('13C', c13_conc))
    diamond.add_isotopes(('14C', p1_conc))  # these are replaced with P1s later
    diamond.zdir = zdir  # set z direction of the defect

    # generate supercell
    atoms = diamond.gen_supercell(size, seed=seed,
                            remove=[('C', [0., 0, 0]), ('C', [0.5, 0.5, 0.5])],  # remove NV carbons IF they're there
                            add=[('14N', [0.5, 0.5, 0.5]), ])  # add NV nitrogen nuclei (electron added later)
    print("disregard user warning about 14C unlesss Exception is thrown")
    atoms.add_type(*SPIN_TYPES)
    assert np.allclose(atoms.A, atoms.A.flat[0])  # check that there are no hyperfine or quadrupoles
    assert np.allclose(atoms.Q, atoms.Q.flat[0])

    # add in P1 nuclei and electrons
    mask = (atoms.N == '14C')
    idx = np.where(mask)[0]
    atoms = append_many(atoms, idx, 'e')  # add in electrons where 14Cs are
    idx_14n = np.where(atoms.N == '14N')[0]
    atoms = append_many(atoms, idx_14n, 'e')  # add in electron where NV nitrogen is
    atoms['N'][idx] = '14N'  # add in P1's by replacing the 14Cs with 14Ns
    assert '14C' not in atoms['N']  # make sure no 14Cs left

    if verbose:
        print("\nBuilt supercell.\n")
        out_file = Path(base_out_dir + '/' + run_id) / (run_id + ".txt")
        with out_file.open("a", encoding="utf-8") as f:
            f.write("\nBUILT SUPERCELL:\n")
            f.write(f"# of atoms: {len(atoms)}\n")
            f.write(f"# of 13C: {len(atoms) - 2 * len(idx)}\n")
            f.write(f"# of P1s: {len(idx)}\n")
            f.write(f"atoms: {atoms}\n")
            # for debugging 14C user warning:
            # names = atoms['N'].astype(str)  # make sure they’re strings
            # f.write("\nH" + " ".join(names) + "\n\n")

    return atoms


def get_simulator(atoms, simulator_params):
    """
    Builds a PyCCE Simulator object given a BathCell which is a supercell with an NV center and a bath of P1s.
    :param atoms: The PyCCE BathCell which is the supercell.
    :param simulator_params: The simulator parameters. This must contain at least order, r_bath, and r_dipole.
    :return: The PyCCE Simulator object.
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    # required
    order = int(simulator_params['order'])
    r_bath = float(simulator_params['r_bath'])
    r_dipole = float(simulator_params['r_dipole'])
    # optional / defaults
    p1_hf = simulator_params.get("p1_hf", True)
    polarization = simulator_params.get("polarization", 0)
    nv_position = simulator_params.get("nv_position", [ 0, 0, 0 ])
    alpha = simulator_params.get("alpha", [ 0, 0, 1 ])
    beta = simulator_params.get("beta", [ 0, 1, 0 ])
    debug_hf = simulator_params.get("debug_hf", False)
    verbose = simulator_params.get("verbose", False)
    run_id = simulator_params["run_id"]
    base_out_dir = simulator_params["base_out_dir"]

    # ================= MAIN CODE ================= #
    # get NV center and imap
    nv = pc.CenterArray(spin=1, position=nv_position, D=D, E=E, alpha=alpha, beta=beta)
    imap = p1_hyperfine(atoms, p1_hf)  # get imap for P1 electron-nuclei hyperfine

    # build simulator
    cce_params = dict(
        bath=atoms,
        spin=nv,
        imap=imap,
        order=order,
        r_bath=r_bath,
        r_dipole=r_dipole
    )
    if debug_hf:
        cce_params.pop("imap")
    calc = pc.Simulator(**cce_params)

    # if gamma is nonzero, will polarize the 13Cs in the Simulator's bath
    if polarization != 0:
        gamma = polarization
        polos = np.exp(-(calc.bath.dist() / gamma) ** 2) * 0.5

        for a, pol in zip(calc.bath, polos):
            if a.N != '13C':
                continue  # Skip 14N and electrons
            # Generate density matrix
            dm = np.zeros((2, 2), dtype=np.complex128)
            dm[0, 0] = 0.5 + pol
            dm[1, 1] = 0.5 - pol
            a.state = dm

    if verbose:
        print("\nBuilt simulator.\n")
        out_file = Path(base_out_dir + '/' + run_id) / (run_id + ".txt")
        with out_file.open("a", encoding="utf-8") as f:
            f.write("\n\nBUILT SIMULATOR:\n")
            f.write(f"{calc}\n")

    return calc


def run_sim(calc, compute_params):
    """
    Calculates the coherence function of some PyCCE Simulator object.
    :param calc: The PyCCE Simulator object to run the simulations with.
    :param compute_params: The compute parameters. This must contain at least magnetic_field, pulse_id, time_space, and
    cce_types.
    :return: A numpy array with 3 columns. First is conv, second is gen, third is MC. The number of rows is time_space.
    Each element in the array is L(t) from 0 to 1.
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    # required
    magnetic_field = compute_params['magnetic_field']
    pulses = get_pulses(compute_params['pulse_id'])
    time_space = compute_params['time_space']
    cce_types = compute_params['cce_types']
    # optional / defaults
    n_bath_states = compute_params.get("n_bath_states", 20)
    verbose = compute_params.get("verbose", False)
    run_id = compute_params["run_id"]
    base_out_dir = compute_params["base_out_dir"]
    parallel = compute_params.get("parallel", False)
    parallel_states = compute_params.get("parallel_states", False)

    # ================= MAIN CODE ================= #
    # initialize the array to be returned
    result = np.zeros((len(time_space), 3), dtype=float)

    # run the simulations
    calc_params = dict(
        magnetic_field=magnetic_field,
        pulses=pulses,
        quantity="coherence",
        parallel=parallel,
        parallel_states=parallel_states,
    )
    if 'conv' in cce_types:  # conventional CCE
        if verbose:
            print("\nStarting conventional...\n")
        l_conv = calc.compute(time_space, method='cce', as_delay=False, **calc_params)
        result[:, 0] = np.real(l_conv)
        if verbose:
            print("\nFinished conventional.\n")
            out_file = Path(base_out_dir + '/' + run_id) / (run_id + ".txt")
            with out_file.open("a", encoding="utf-8") as f:
                f.write(f"\nConventional coherence: {l_conv}")
    if 'gen' in cce_types:  # generalized CCE
        l_generalized = calc.compute(time_space, method='gcce',**calc_params)
        result[:, 1] = np.real(l_generalized)
        if verbose:
            print("\nFinished generalized.\n")
            out_file = Path(base_out_dir + '/' + run_id) / (run_id + ".txt")
            with out_file.open("a", encoding="utf-8") as f:
                f.write(f"\nGeneralized coherence: {l_generalized}")
    if 'MC' in cce_types:  # generalized CCE with random sampling of bath states
        l_generalized_mc = calc.compute(time_space, nbstates=n_bath_states, method='gcce', seed=UNIVERSAL_SEED,
                                        **calc_params)
        result[:, 2] = np.real(l_generalized_mc)
        if verbose:
            print("\nFinished generalized with MC.\n")
            out_file = Path(base_out_dir + '/' + run_id) / (run_id + ".txt")
            with out_file.open("a", encoding="utf-8") as f:
                f.write(f"\nGeneralized with MC coherence: {l_generalized_mc}")

    return result


def run_ensemble(model_params, simulator_params, compute_params, ensemble_params):
    """
    Runs many simulations each with a unique bath configuration. A glorified for loop of the function run_sim.
    :param model_params: see the function get_model
    :param simulator_params: see the function get_simulator
    :param compute_params: see the function run_sim
    :param ensemble_params: The ensemble parameters. This must contain at least n_runs and average_type.
    :return: A dictionary of length 3, of 2D numpy arrays. The keys are the 3 cce_types; if a cce_type (e.g. 'MC') is
    not in cce_types, then its value is None. The 2D arrays have shape (timesteps, ensemble_size); that is, each column
    is one experiment, so the number of columns is the ensemble size. And the number of rows is just how many steps in
    the time space. Each element is L(t) from 0 to 1.
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    # required
    ensemble_size = ensemble_params['ensemble_size']
    # optional / defaults
    verbose = ensemble_params.get("verbose", False)
    run_id = ensemble_params["run_id"]
    base_out_dir = ensemble_params["base_out_dir"]

    # ================= MAIN CODE ================= #
    results = []

    for i in range(ensemble_size):
        seed = np.random.randint(0, 2 ** 32, dtype='uint32')  # new seed each loop
        new_model_params = dict(model_params)  # shallow copy so we can tweak per-run
        new_model_params["seed"] = seed

        supercell = get_model(new_model_params)
        simulator = get_simulator(supercell, simulator_params)
        coherence = run_sim(simulator, compute_params)
        results.append(coherence)

        if verbose:
            print(f"\n\nFinished run {i+1}.")
            out_file = Path(base_out_dir + '/' + run_id) / (run_id + ".txt")
            with out_file.open("a", encoding="utf-8") as f:
                f.write(f"\n\nFinished run {i+1}. More information below.\n")
                # f.write(f"\nSupercell:{supercell}")
                # f.write(f"\nSimulator:{simulator}")
                # f.write(f"\nCoherence:{coherence}")
                # f.write(f"\nModel params:{model_params}")
                # f.write(f"\nSimulator params:{simulator_params}")
                # f.write(f"\nCompute params:{compute_params}")
                # f.write(f"\nEnsemble params:{ensemble_params}")

    results_arr = np.stack(results, axis=2)
    final = {
        "conv": results_arr[:, 0, :] if "conv" in compute_params["cce_types"] else None,
        "gen": results_arr[:, 1, :] if "gen" in compute_params["cce_types"] else None,
        "MC": results_arr[:, 2, :] if "MC" in compute_params["cce_types"] else None,
    }

    return final
