"""Training JODO on GEOM-Drugs"""

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    config.exp_type = 'vpsde_edge'
    config.pred_edge = True
    config.only_2D = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.root = 'data/geom'
    data.name = 'GeomDrug'
    data.processed_file = 'data_geom_drug_1.pt'
    data.transform = 'EdgeCom'
    data.collate = 'collate_edge'
    data.info_name = 'geom_with_h_1'
    data.num_workers = 16

    data.compress_edge = True
    data.centered = True  # center with 0
    data.include_aromatic = True
    data.atom_types = 16
    data.bond_types = 5
    data.fc_scale = [-2., 3.]
    data.max_node = 181

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'  # 'discrete_poly', 'linear', 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    # model
    config.model = model = ml_collections.ConfigDict()

    # # common model parameters
    model.name = 'DGT_concat'
    model.pred_data = True  # False: noise_prediction; True: data_prediction
    model.include_fc_charge = True  # include formal charges of atoms
    model.normalize_factors = '1, 4, 4, 1'  # normalize factors of position, atom types, formal charges, edge types
    model.ema_decay = 0.999
    model.edge_ch = 3  # input edge channels
    model.nf = 256  # node hidden channels
    model.n_layers = 10  # number of blocks
    model.n_heads = 16  # number of attention heads
    model.dropout = 0.1
    model.cond_time = True
    model.dist_gbf = True
    model.gbf_name = 'CondGaussianLayer'
    model.self_cond = True
    model.self_cond_type = 'ori'  # 'ori', 'clamp'

    model.edge_quan_th = 0.  # edge quantization threshold
    model.n_extra_heads = 2  # attention heads from adjacency matrices
    model.CoM = True  # keep each layer output in CoM
    model.mlp_ratio = 4  # FFN channel ratio
    model.spatial_cut_off = 3.  # distance threshold for spatial adjacency matrix
    model.softmax_inf = True
    model.trans_name = 'TransMixLayer'

    # # loss function
    model.loss_weights = '1, 0.25, 0.1'
    model.noise_align = True  # Kabsch align before loss

    # training
    config.training = training = ml_collections.ConfigDict()
    training.reduce_mean = False
    training.batch_size = 16
    training.eval_batch_size = 16
    training.eval_samples = 128
    training.log_freq = 500

    training.n_iters = 1500000
    training.snapshot_freq = 50000
    # # store additional checkpoints for preemption (meta)
    training.snapshot_freq_for_preemption = 10000
    # # produce samples at each snapshot.
    training.snapshot_sampling = True

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 100000
    optim.grad_clip = 20.
    optim.disable_grad_log = True

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'ancestral'  # 'ancestral' or 'fast'
    sampling.steps = 1000
    sampling.vis_row = 4
    sampling.vis_col = 4
    sampling.dpm_solver_method = 'singlestep_fixed'  # 'singlestep_fixed' or 'multistep'
    sampling.dpm_solver_order = 2  # 2, 3 for single step; 2 for multistep

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.enable_sampling = True
    evaluate.batch_size = 1000  # generation batch size
    evaluate.num_samples = 10000  # number of samples for evaluation
    evaluate.begin_ckpt = 30
    evaluate.end_ckpt = 30
    evaluate.ckpts = ''  # eg. '30'; '25, 30'
    evaluate.save_graph = False  # save generation samples
    evaluate.sub_geometry = True

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
