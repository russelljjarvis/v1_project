import numpy as np
import pandas as pd
import json
from pathlib import Path
from sonata.circuit import File
from sonata.reports.spike_trains import SpikeTrains
import pygenn
import matplotlib.pyplot as plt

GLIF3 = pygenn.genn_model.create_custom_neuron_class(
    "GLIF3",
    param_names=["C", "G", "El", "spike_cut_length", "th_inf", "ASC_length", "V_reset"],
    var_name_types=[("V", "double"), ("refractory_countdown", "int")],
    extra_global_params=[
        ("ASC", "scalar*"),
        ("k", "scalar*"),
        ("asc_amp_array", "scalar*"),
        ("r", "scalar*"),
    ],
    sim_code="""
    double sum_of_ASC = 0.0;
    
    // Sum after spike currents
    for (int ii=0; ii<$(ASC_length); ii++){
        int idx = $(id)*((int)$(ASC_length))+ii;
        sum_of_ASC += $(ASC)[idx];
        }
    // Voltage
    if ($(refractory_countdown) > 0) {
        $(V) += 0.0;
    }
    else {
        $(V)+=1/$(C)*($(Isyn)+sum_of_ASC-$(G)*($(V)-$(El)))*DT;
    }
    // ASCurrents
    if ($(refractory_countdown) > 0) {
        for (int ii=0; ii<$(ASC_length); ii++){
            int idx = $(id)*((int)$(ASC_length))+ii;
            $(ASC)[idx] += 0.0;
            }
    }
    else {
        for (int ii=0; ii<$(ASC_length); ii++){
            int idx = $(id)*((int)$(ASC_length))+ii;
            $(ASC)[idx] = $(ASC)[idx] * exp(-$(k)[idx]*DT);
            }
    }
    // Decrement refractory_countdown by 1; Do not decrement past -1
    if ($(refractory_countdown) > -1) {
        $(refractory_countdown) -= 1;
    }
    """,
    threshold_condition_code="$(V) > $(th_inf)",
    reset_code="""
    $(V)=$(V_reset);
    for (int ii=0; ii<$(ASC_length); ii++){
        int idx = $(id)*((int)$(ASC_length))+ii;
        $(ASC)[idx] = $(asc_amp_array)[idx] + $(ASC)[idx] * $(r)[idx] * exp(-($(k)[idx] * DT * $(spike_cut_length)));
        }
    $(refractory_countdown) = $(spike_cut_length);
    """,
)

psc_Alpha = pygenn.genn_model.create_custom_postsynaptic_class(
    class_name="Alpha",
    decay_code="""
    $(x) = exp(-DT/$(tau)) * ((DT * $(inSyn) * exp(1.0f) / $(tau)) + $(x));
    $(inSyn)*=exp(-DT/$(tau));
    """,
    apply_input_code="""
    $(Isyn) += $(x);     
""",
    var_name_types=[("x", "scalar")],
    param_names=[("tau")],
)


def spikes_list_to_start_end_times(spikes_list):

    spike_counts = [len(n) for n in spikes_list]

    # Get start and end indices of each spike sources section
    end_spike = np.cumsum(spike_counts)
    start_spike = np.empty_like(end_spike)
    start_spike[0] = 0
    start_spike[1:] = end_spike[0:-1]

    spike_times = np.hstack(spikes_list)
    return start_spike, end_spike, spike_times


def get_dynamics_params(
    node_types_df, dynamics_base_dir, sim_config, node_dict, model_name
):
    dynamics_file = node_types_df[node_types_df["model_name"] == model_name][
        "dynamics_params"
    ].iloc[0]
    dynamics_path = Path(dynamics_base_dir, dynamics_file)
    with open(dynamics_path) as f:
        old_dynamics_params = json.load(f)
    num_neurons = len(node_dict[model_name])

    dynamics_params_renamed = {
        "C": old_dynamics_params["C_m"] / 1000,  # pF -> nF
        "G": old_dynamics_params["g"] / 1000,  # nS -> uS
        "El": old_dynamics_params["E_L"],
        "th_inf": old_dynamics_params["V_th"],
        "dT": sim_config["run"]["dt"],
        "V": old_dynamics_params["V_m"],
        "spike_cut_length": round(
            old_dynamics_params["t_ref"] / sim_config["run"]["dt"]
        ),
        "refractory_countdown": -1,
        "k": 1
        / np.repeat(
            old_dynamics_params["asc_decay"], num_neurons
        ).ravel(),  # TODO: Reciprocal?
        "r": np.repeat([1.0, 1.0], num_neurons).ravel(),
        "ASC": np.repeat(old_dynamics_params["asc_init"], num_neurons).ravel()
        / 1000,  # pA -> nA
        "asc_amp_array": np.repeat(old_dynamics_params["asc_amps"], num_neurons).ravel()
        / 1000,  # pA -> nA
        "ASC_length": 2,
        "V_reset": old_dynamics_params["V_reset"],  # BMTK rounds to 3rd decimal
        "tau": old_dynamics_params["tau_syn"][0],  # Always 0th port?
    }

    return dynamics_params_renamed, num_neurons


def construct_populations(
    model,
    pop_dict,
    all_model_names,
    node_dict,
    dynamics_base_dir,
    node_types_df,
    neuron_class,
    sim_config,
):
    for i, model_name in enumerate(all_model_names):

        dynamics_params_renamed, num_neurons = get_dynamics_params(
            node_types_df, dynamics_base_dir, sim_config, node_dict, model_name
        )

        params = {k: dynamics_params_renamed[k] for k in neuron_class.get_param_names()}
        init = {k: dynamics_params_renamed[k] for k in ["V", "refractory_countdown"]}

        pop_dict[model_name] = model.add_neuron_population(
            pop_name=model_name,
            num_neurons=num_neurons,
            neuron=neuron_class,
            param_space=params,
            var_space=init,
        )

        # Assign extra global parameter values
        for k in pop_dict[model_name].extra_global_params.keys():
            pop_dict[model_name].set_extra_global_param(k, dynamics_params_renamed[k])

        print("{} population added to model.".format(model_name))
    return pop_dict


def construct_synapses(
    model,
    syn_dict,
    pop1,
    pop2,
    all_edge_type_ids,
    all_nsyns,
    edge_df,
    syn_df,
    sim_config,
    dynamics_params,
):
    for edge_type_id in all_edge_type_ids:
        for nsyns in all_nsyns:

            # Filter by source, target, edge_type_id, and nsyns
            src = edge_df[~edge_df["source_" + pop1].isnull()]
            src_tgt = src[~src["target_" + pop2].isnull()]
            src_tgt_id = src_tgt[src_tgt["edge_type_id"] == edge_type_id]
            src_tgt_id_nsyns = src_tgt_id[src_tgt_id["nsyns"] == nsyns]

            # Convert to list for GeNN
            s_list = src_tgt_id_nsyns["source_" + pop1].tolist()
            t_list = src_tgt_id_nsyns["target_" + pop2].tolist()

            # Skip if no synapses (typically only 1 edge_type_id relevant for each source)
            if len(s_list) == 0:
                # print(
                #     "No synapses found for {} -> {} with edge type id={}".format(
                #         pop1,
                #         pop2,
                #         edge_type_id,
                #     )
                # )
                continue

            # Test correct assignment
            assert np.all(~src_tgt_id_nsyns.isnull(), axis=0).sum() == 6

            # Get delay and weight specific to the edge_type_id
            delay_steps = round(
                syn_df[syn_df["edge_type_id"] == edge_type_id]["delay"].iloc[0]
                / sim_config["run"]["dt"]
            )  # delay (ms) -> delay (steps)
            weight = (
                syn_df[syn_df["edge_type_id"] == edge_type_id]["syn_weight"].iloc[0]
                / 1e3
                * nsyns
            )  # nS -> uS; multiply by number of synapses

            s_ini = {"g": weight}
            psc_Alpha_params = {"tau": dynamics_params["tau"]}  # TODO: Always 0th port?
            psc_Alpha_init = {"x": 0.0}

            synapse_group_name = pop1 + "_to_" + pop2 + "_nsyns_" + str(nsyns)
            syn_dict[synapse_group_name] = model.add_synapse_population(
                pop_name=synapse_group_name,
                matrix_type="SPARSE_GLOBALG_INDIVIDUAL_PSM",
                delay_steps=delay_steps,
                source=pop1,
                target=pop2,
                w_update_model="StaticPulse",
                wu_param_space={},
                wu_var_space=s_ini,
                wu_pre_var_space={},
                wu_post_var_space={},
                postsyn_model=psc_Alpha,
                ps_param_space=psc_Alpha_params,
                ps_var_space=psc_Alpha_init,
            )
            syn_dict[synapse_group_name].set_sparse_connections(
                np.array(s_list), np.array(t_list)
            )
            print(
                "Synapses added for {} -> {} with edge type id={} and nsyns={}".format(
                    pop1, pop2, edge_type_id, nsyns
                )
            )
    return syn_dict


def construct_id_conversion_df(
    edges, all_model_names, source_node_to_pop_idx_dict, target_node_to_pop_idx_dict
):
    edges_for_df = []
    for e in edges:
        e_dict = {}

        # Add node_ids
        e_dict["source_node_id"] = e.source_node_id
        e_dict["target_node_id"] = e.target_node_id

        # Populate empty indices for each population
        for m in all_model_names:
            e_dict["source_" + m] = pd.NA
            e_dict["target_" + m] = pd.NA

        # Populate actual dicts
        [m_name, idx] = source_node_to_pop_idx_dict[e.source_node_id]
        e_dict["source_" + m_name] = idx

        [m_name, idx] = target_node_to_pop_idx_dict[e.target_node_id]
        e_dict["target_" + m_name] = idx

        # Add edge type id, which is used to get correct synaptic weight/delay
        # TODO: Is dynamics_params e.g. e2i.json used?
        e_dict["edge_type_id"] = e.edge_type_id

        # Add number of synapses
        e_dict["nsyns"] = e["nsyns"]

        edges_for_df.append(e_dict)
    v1_edge_df = pd.DataFrame(edges_for_df)
    return v1_edge_df
