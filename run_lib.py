import os
import torch
import logging
import numpy as np
import pdb
import random
import pickle
# from torch.utils import tensorboard

from datasets import get_dataset, inf_iterator, get_dataloader
from models.ema import ExponentialMovingAverage
import losses
from utils import *
from evaluation import *
import visualize
from models import *
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn, get_cond_sampling_eval_fn, get_cond_multi_sampling_eval_fn
from cond_gen import *


def set_random_seed(config):
    seed = config.seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def vpsde_edge_train(config, workdir):
    """Runs the training pipeline with VPSDE for geometry graphs."""

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Build dataset and dataloader
    train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    train_loader, val_loader, test_loader = get_dataloader(train_ds, val_ds, test_ds, config)

    train_iter = inf_iterator(train_loader)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    num_train_steps = config.training.n_iters

    if initial_step == 0:
        logging.info(config)

    # get loss step optimizer
    optimize_fn = losses.optimization_manager(config)
    # change step fn
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config)

    # Build sampling functions
    if config.training.snapshot_sampling:
        # change sampling fn
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.training.eval_batch_size,
                                      config.training.eval_samples, inverse_scaler)

    # Build evaluation metric
    EDM_metric = get_edm_metric(dataset_info)
    EDM_metric_2D = get_2D_edm_metric(dataset_info)
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]
    fcd_metric = get_fcd_metric(test_mols, n_jobs=32, device=config.device, batch_size=1000)

    # Training iterations
    for step in range(initial_step, num_train_steps + 1):
        batch = next(train_iter)

        # Execute one training step
        loss = train_step_fn(state, batch)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate, evaluate and save samples
            if config.training.snapshot_sampling:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

                # Wrapper EDM sampling
                processed_mols = sampling_fn(model)

                # EDM evaluation metrics
                stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
                logging.info("step: %d, n_mol: %d, 3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                             "complete: %.4f, unique & valid: %.4f" % (
                             step, len(sample_rdmols), stability_res['atom_stable'],
                             stability_res['mol_stable'], rdkit_res['Validity'],
                             rdkit_res['Complete'], rdkit_res['Unique']))

                # FCD metric
                fcd_res = fcd_metric(sample_rdmols)
                logging.info("3D FCD: %.4f" % (fcd_res['FCD']))

                # 2D evaluations
                stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
                logging.info("step: %d, n_mol: %d, 2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                             "complete: %.4f, unique & valid: %.4f" % (
                                 step, len(sample_rdmols), stability_res['atom_stable'],
                                 stability_res['mol_stable'], rdkit_res['Validity'],
                                 rdkit_res['Complete'], rdkit_res['Unique']))
                fcd_res = fcd_metric(complete_rdmols)
                logging.info("2D FCD: %.4f" % (fcd_res['FCD']))

                ema.restore(model.parameters())

                # Visualization
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                if not os.path.exists(this_sample_dir):
                    os.makedirs(this_sample_dir)

                # change `sample_rdmols` to `complete_rdmols`?
                visualize.visualize_mols(sample_rdmols, this_sample_dir, config)


def vpsde_edge_evaluate(config, workdir, eval_folder="eval"):
    """Runs the evaluation pipeline with VPSDE for geometry graphs."""
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    # scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.eval.batch_size,
                                      config.eval.num_samples, inverse_scaler)

    # Obtain train dataset and eval dataset
    train_mols = [train_ds[i].rdmol for i in range(len(train_ds))]
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build evaluation metrics
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    mose_metric = get_moses_metrics(test_mols, n_jobs=32, device=config.device)
    if config.eval.sub_geometry:
        sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, config.data.root)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        if config.eval.enable_sampling:
            logging.info('Sampling -- ckpt: %d' % (ckpt,))
            # Wrapper EDM sampling
            processed_mols = sampling_fn(model)

            # EDM evaluation metrics
            stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
            logging.info('Number of molecules: %d' % len(sample_rdmols))
            logging.info("3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                         " unique & valid: %.4f, novelty: %.4f" % (stability_res['atom_stable'],
                                                                   stability_res['mol_stable'], rdkit_res['Validity'],
                                                                   rdkit_res['Complete'], rdkit_res['Unique'],
                                                                   rdkit_res['Novelty']))

            # Mose evaluation metrics
            mose_res = mose_metric(sample_rdmols)
            logging.info("3D FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                         mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))

            # 2D evaluation metrics
            stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
            logging.info("2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                         " unique & valid: %.4f, novelty: %.4f" % (stability_res['atom_stable'],
                                                                   stability_res['mol_stable'], rdkit_res['Validity'],
                                                                   rdkit_res['Complete'], rdkit_res['Unique'],
                                                                   rdkit_res['Novelty']))
            mose_res = mose_metric(complete_rdmols)
            logging.info("2D FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                         mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))

            logging.info("Mean QED: %.4f, MCF: %.4f, SAS: %.4f, logP: %.4f, MW: %.4f" % (mose_res['QED'],
                          mose_res['Filters'], mose_res['SA'], mose_res['logP'], mose_res['weight']))

            # save sample mols
            if config.eval.save_graph:
                sampler_name = config.sampling.method
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}_{}.pkl".format(ckpt, config.seed))
                with open(graph_file, "wb") as f:
                    pickle.dump(complete_rdmols, f)

            # Substructure Geometry MMD
            if config.eval.sub_geometry:
                sub_geo_mmd_res = sub_geo_mmd_metric(complete_rdmols)
                logging.info("Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
                    sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
                    sub_geo_mmd_res['dihedral_angle_mean']))
                ## bond length
                bond_length_str = ''
                for sym in dataset_info['top_bond_sym']:
                    bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
                logging.info(bond_length_str)
                ## bond angle
                bond_angle_str = ''
                for sym in dataset_info['top_angle_sym']:
                    bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(bond_angle_str)
                ## dihedral angle
                dihedral_angle_str = ''
                for sym in dataset_info['top_dihedral_sym']:
                    dihedral_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(dihedral_angle_str)


def vpsde_train(config, workdir):
    """Runs the training pipeline with VPSDE for graphs or point clouds"""

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Build dataset and dataloader
    train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    train_loader, val_loader, test_loader = get_dataloader(train_ds, val_ds, test_ds, config)

    train_iter = inf_iterator(train_loader)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    num_train_steps = config.training.n_iters

    if initial_step == 0:
        logging.info(config)

    # get loss step optimizer
    optimize_fn = losses.optimization_manager(config)
    # change step fn
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config)

    # Build sampling functions
    if config.training.snapshot_sampling:
        # change sampling fn
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.training.eval_batch_size,
                                      config.training.eval_samples, inverse_scaler)

    # Build evaluation metric
    EDM_metric = get_edm_metric(dataset_info)
    EDM_metric_2D = get_2D_edm_metric(dataset_info)
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]
    fcd_metric = get_fcd_metric(test_mols, n_jobs=32, device=config.device, batch_size=1000)

    # Training iterations
    for step in range(initial_step, num_train_steps + 1):
        batch = next(train_iter)
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Visualization
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            if not os.path.exists(this_sample_dir):
                os.makedirs(this_sample_dir)

            # Generate, evaluate and save samples
            if config.training.snapshot_sampling:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

                # Wrapper EDM sampling
                processed_mols = sampling_fn(model)

                if not config.only_2D:
                    # 3D positions evaluation
                    ## EDM evaluation metrics
                    stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
                    logging.info("step: %d, n_mol: %d, atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                    "complete: %.4f, unique & valid: %.4f"%(step, len(sample_rdmols), stability_res['atom_stable'],
                                                            stability_res['mol_stable'], rdkit_res['Validity'],
                                                            rdkit_res['Complete'], rdkit_res['Unique']))

                    ## FCD metric
                    fcd_res = fcd_metric(sample_rdmols)
                    logging.info("FCD: %.4f" % (fcd_res['FCD']))
                    visualize.visualize_mols(sample_rdmols, this_sample_dir, config)
                else:
                    # 2D evaluation
                    stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
                    logging.info("step: %d, n_mol: %d, 2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                                 "complete: %.4f, unique & valid: %.4f" % (
                                 step, len(complete_rdmols), stability_res['atom_stable'],
                                 stability_res['mol_stable'], rdkit_res['Validity'],
                                 rdkit_res['Complete'], rdkit_res['Unique']))
                    fcd_res = fcd_metric(complete_rdmols)
                    logging.info("2D FCD: %.4f" % (fcd_res['FCD']))
                    visualize.visualize_mols(complete_rdmols, this_sample_dir, config, check_valid=True)

                ema.restore(model.parameters())


def vpsde_evaluate(config, workdir, eval_folder="eval"):
    """Runs the evaluation pipeline with VPSDE for graphs or point clouds"""
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    # scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.eval.batch_size,
                                      config.eval.num_samples, inverse_scaler)

    # Obtain train dataset and eval dataset
    train_mols = [train_ds[i].rdmol for i in range(len(train_ds))]
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build evaluation metrics
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    mose_metric = get_moses_metrics(test_mols, n_jobs=32, device=config.device)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        if config.eval.enable_sampling:
            logging.info('Sampling -- ckpt: %d' % (ckpt,))

            # Wrapper EDM sampling
            processed_mols = sampling_fn(model)

            if not config.only_2D:
                # EDM evaluation metrics
                stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
                logging.info('Number of molecules: %d' % len(sample_rdmols))
                logging.info("atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                             " unique & valid: %.4f, novelty: %.4f" % (stability_res['atom_stable'],
                             stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete'], rdkit_res['Unique'],
                             rdkit_res['Novelty']))

                # Mose evaluation metrics
                mose_res = mose_metric(sample_rdmols)
                logging.info("FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                             mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))
            else:
                # pure 2D evaluation metrics
                stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
                logging.info("2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                             " unique & valid: %.4f, novelty: %.4f" % (stability_res['atom_stable'],
                             stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete'],
                             rdkit_res['Unique'], rdkit_res['Novelty']))
                mose_res = mose_metric(complete_rdmols)
                logging.info("2D FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                             mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))

                logging.info("Mean QED: %.4f, MCF: %.4f, SAS: %.4f, logP: %.4f, MW: %.4f" % (mose_res['QED'],
                             mose_res['Filters'], mose_res['SA'], mose_res['logP'], mose_res['weight']))

            # save sample mols
            if config.eval.save_graph:
                sampler_name = config.sampling.method
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}_{}.pkl".format(ckpt, config.seed))
                with open(graph_file, "wb") as f:
                    if not config.only_2D:
                        pickle.dump(sample_rdmols, f)
                    else:
                        pickle.dump(complete_rdmols, f)


def vpsde_edge_cond_train(config, workdir):
    """Runs the training pipeline with VPSDE for geometry graphs with additional quantum property conditioning."""

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Build dataset and dataloader
    _, second_train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    prop2idx = dataset_info['prop2idx']
    ## one property
    prop2idx_sub = {
        config.cond_property: prop2idx[config.cond_property]
    }
    prop_norms = val_ds.compute_property_mean_mad(prop2idx_sub)
    prop_dist = DistributionProperty(second_train_ds, prop2idx_sub, normalizer=prop_norms)
    train_loader, val_loader, test_loader = get_dataloader(second_train_ds, val_ds, test_ds, config)
    train_iter = inf_iterator(train_loader)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    num_train_steps = config.training.n_iters

    if initial_step == 0:
        logging.info(config)

    # get loss step optimizer
    optimize_fn = losses.optimization_manager(config)
    # change step fn
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config, prop_norms)

    # Build sampling functions
    if config.training.snapshot_sampling:
        # change sampling fn
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.training.eval_batch_size,
                                      config.training.eval_samples, inverse_scaler, prop_dist=prop_dist)

    # Build evaluation metric
    EDM_metric = get_edm_metric(dataset_info)
    EDM_metric_2D = get_2D_edm_metric(dataset_info)
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]
    fcd_metric = get_fcd_metric(test_mols, n_jobs=32, device=config.device, batch_size=1000)

    # Training iterations
    for step in range(initial_step, num_train_steps + 1):
        batch = next(train_iter)

        # Execute one training step
        loss = train_step_fn(state, batch)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate, evaluate and save samples
            if config.training.snapshot_sampling:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

                # Wrapper EDM sampling
                processed_mols = sampling_fn(model)

                # EDM evaluation metrics
                stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
                logging.info("step: %d, n_mol: %d, 3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                             "complete: %.4f, unique & valid: %.4f" % (
                             step, len(sample_rdmols), stability_res['atom_stable'],
                             stability_res['mol_stable'], rdkit_res['Validity'],
                             rdkit_res['Complete'], rdkit_res['Unique']))

                # FCD metric
                fcd_res = fcd_metric(sample_rdmols)
                logging.info("3D FCD: %.4f" % (fcd_res['FCD']))

                # 2D evaluations
                stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
                logging.info("step: %d, n_mol: %d, 2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                             "complete: %.4f, unique & valid: %.4f" % (
                                 step, len(sample_rdmols), stability_res['atom_stable'],
                                 stability_res['mol_stable'], rdkit_res['Validity'],
                                 rdkit_res['Complete'], rdkit_res['Unique']))
                fcd_res = fcd_metric(complete_rdmols)
                logging.info("2D FCD: %.4f" % (fcd_res['FCD']))

                ema.restore(model.parameters())

                # Visualization
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                if not os.path.exists(this_sample_dir):
                    os.makedirs(this_sample_dir)

                visualize.visualize_mols(sample_rdmols, this_sample_dir, config)


def vpsde_edge_cond_evaluate(config, workdir, eval_folder="eval"):
    """Runs the evaluation pipeline with VPSDE for geometry graphs with additional quantum property conditioning."""

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    # train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)
    _, second_train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    prop2idx = dataset_info['prop2idx']

    ## one property
    prop2idx_sub = {
        config.cond_property: prop2idx[config.cond_property]
    }
    prop_norms = val_ds.compute_property_mean_mad(prop2idx_sub)
    prop_dist = DistributionProperty(second_train_ds, prop2idx_sub, normalizer=prop_norms)

    # Load property prediction model for evaluation
    property_path = os.path.join(config.data.root, 'property_classifier', f'evaluate_{config.cond_property}')
    classifier_path = os.path.join(property_path, 'best_checkpoint.npy')
    args_classifier_path = os.path.join(property_path, 'args.pickle')
    classifier = get_classifier(classifier_path, args_classifier_path).to(config.device)
    classifier = torch.nn.DataParallel(classifier)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    # scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_cond_sampling_eval_fn(config, noise_scheduler, nodes_dist, config.eval.batch_size,
                                                config.eval.num_samples, inverse_scaler, prop_dist=prop_dist,
                                                prop_norm=prop_norms)

    # Obtain train dataset and eval dataset
    train_mols = [second_train_ds[i].rdmol for i in range(len(second_train_ds))]
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build evaluation metrics
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    mose_metric = get_moses_metrics(test_mols, n_jobs=32, device=config.device)
    if config.eval.sub_geometry:
        sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, config.data.root)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        if config.eval.enable_sampling:
            logging.info('Sampling -- ckpt: %d' % (ckpt,))
            # Wrapper EDM sampling
            processed_mols, MAE_loss = sampling_fn(model, classifier)
            logging.info(f"{config.cond_property} MAE: %.4f" % MAE_loss)

            # EDM evaluation metrics
            stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
            logging.info('Number of molecules: %d' % len(sample_rdmols))
            logging.info("3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                         " unique & valid: %.4f, novelty: %.4f" % (stability_res['atom_stable'],
                                                                   stability_res['mol_stable'], rdkit_res['Validity'],
                                                                   rdkit_res['Complete'], rdkit_res['Unique'],
                                                                   rdkit_res['Novelty']))

            # Mose evaluation metrics
            mose_res = mose_metric(sample_rdmols)
            logging.info("3D FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                         mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))

            # 2D evaluation metrics
            stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
            logging.info("2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                         " unique & valid: %.4f, novelty: %.4f" % (stability_res['atom_stable'],
                                                                   stability_res['mol_stable'], rdkit_res['Validity'],
                                                                   rdkit_res['Complete'], rdkit_res['Unique'],
                                                                   rdkit_res['Novelty']))
            mose_res = mose_metric(complete_rdmols)
            logging.info("2D FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                         mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))

            # save sample mols
            if config.eval.save_graph:
                sampler_name = config.sampling.method
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}_{}.pkl".format(ckpt, config.seed))
                with open(graph_file, "wb") as f:
                    pickle.dump(complete_rdmols, f)

            # Substructure Geometry MMD
            if config.eval.sub_geometry:
                sub_geo_mmd_res = sub_geo_mmd_metric(complete_rdmols)
                logging.info("Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
                    sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
                    sub_geo_mmd_res['dihedral_angle_mean']))
                ## bond length
                bond_length_str = ''
                for sym in dataset_info['top_bond_sym']:
                    bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
                logging.info(bond_length_str)
                ## bond angle
                bond_angle_str = ''
                for sym in dataset_info['top_angle_sym']:
                    bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(bond_angle_str)
                ## dihedral angle
                dihedral_angle_str = ''
                for sym in dataset_info['top_dihedral_sym']:
                    dihedral_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(dihedral_angle_str)


def vpsde_edge_cond_multi_train(config, workdir):
    """Runs the training pipeline with VPSDE for geometry graphs with additional quantum property conditioning.
    Two conditional property"""

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Build dataset and dataloader
    _, second_train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    prop2idx = dataset_info['prop2idx']
    ## two property
    prop2idx_sub = {
        config.cond_property1: prop2idx[config.cond_property1],
        config.cond_property2: prop2idx[config.cond_property2]
    }
    prop_norms = val_ds.compute_property_mean_mad(prop2idx_sub)
    prop_dist = DistributionProperty(second_train_ds, prop2idx_sub, normalizer=prop_norms)
    train_loader, val_loader, test_loader = get_dataloader(second_train_ds, val_ds, test_ds, config)
    train_iter = inf_iterator(train_loader)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    num_train_steps = config.training.n_iters

    if initial_step == 0:
        logging.info(config)

    # get loss step optimizer
    optimize_fn = losses.optimization_manager(config)
    # change step fn
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config, prop_norms)

    # Build sampling functions
    if config.training.snapshot_sampling:
        # change sampling fn
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.training.eval_batch_size,
                                      config.training.eval_samples, inverse_scaler, prop_dist=prop_dist)

    # Build evaluation metric
    EDM_metric = get_edm_metric(dataset_info)
    EDM_metric_2D = get_2D_edm_metric(dataset_info)
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]
    fcd_metric = get_fcd_metric(test_mols, n_jobs=32, device=config.device, batch_size=1000)

    # Training iterations
    for step in range(initial_step, num_train_steps + 1):
        batch = next(train_iter)

        # Execute one training step
        loss = train_step_fn(state, batch)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate, evaluate and save samples
            if config.training.snapshot_sampling:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

                # Wrapper EDM sampling
                processed_mols = sampling_fn(model)

                # EDM evaluation metrics
                stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
                logging.info("step: %d, n_mol: %d, 3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                             "complete: %.4f, unique & valid: %.4f" % (
                             step, len(sample_rdmols), stability_res['atom_stable'],
                             stability_res['mol_stable'], rdkit_res['Validity'],
                             rdkit_res['Complete'], rdkit_res['Unique']))

                # FCD metric
                fcd_res = fcd_metric(sample_rdmols)
                logging.info("3D FCD: %.4f" % (fcd_res['FCD']))

                # 2D evaluations
                stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
                logging.info("step: %d, n_mol: %d, 2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                             "complete: %.4f, unique & valid: %.4f" % (
                                 step, len(sample_rdmols), stability_res['atom_stable'],
                                 stability_res['mol_stable'], rdkit_res['Validity'],
                                 rdkit_res['Complete'], rdkit_res['Unique']))
                fcd_res = fcd_metric(complete_rdmols)
                logging.info("2D FCD: %.4f" % (fcd_res['FCD']))

                ema.restore(model.parameters())

                # Visualization
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                if not os.path.exists(this_sample_dir):
                    os.makedirs(this_sample_dir)

                visualize.visualize_mols(sample_rdmols, this_sample_dir, config)


def vpsde_edge_cond_multi_evaluate(config, workdir, eval_folder="eval"):
    """Runs the evaluation pipeline with VPSDE for geometry graphs with additional quantum property conditioning."""

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    # train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)
    _, second_train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    prop2idx = dataset_info['prop2idx']

    ## two property
    prop2idx_sub = {
        config.cond_property1: prop2idx[config.cond_property1],
        config.cond_property2: prop2idx[config.cond_property2]
    }
    prop_norms = val_ds.compute_property_mean_mad(prop2idx_sub)
    prop_dist = DistributionProperty(second_train_ds, prop2idx_sub, normalizer=prop_norms)

    # Load property prediction model for evaluation
    property_path1 = os.path.join(config.data.root, 'property_classifier', f'evaluate_{config.cond_property1}')
    classifier_path1 = os.path.join(property_path1, 'best_checkpoint.npy')
    args_classifier_path1 = os.path.join(property_path1, 'args.pickle')
    classifier1 = get_classifier(classifier_path1, args_classifier_path1).to(config.device)
    classifier1 = torch.nn.DataParallel(classifier1)

    property_path2 = os.path.join(config.data.root, 'property_classifier', f'evaluate_{config.cond_property2}')
    classifier_path2 = os.path.join(property_path2, 'best_checkpoint.npy')
    args_classifier_path2 = os.path.join(property_path2, 'args.pickle')
    classifier2 = get_classifier(classifier_path2, args_classifier_path2).to(config.device)
    classifier2 = torch.nn.DataParallel(classifier2)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    # scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_cond_multi_sampling_eval_fn(config, noise_scheduler, nodes_dist, config.eval.batch_size,
                                                      config.eval.num_samples, inverse_scaler, prop_dist=prop_dist,
                                                      prop_norm=prop_norms)

    # Obtain train dataset and eval dataset
    train_mols = [second_train_ds[i].rdmol for i in range(len(second_train_ds))]
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build evaluation metrics
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    mose_metric = get_moses_metrics(test_mols, n_jobs=32, device=config.device)
    if config.eval.sub_geometry:
        sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, config.data.root)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        if config.eval.enable_sampling:
            logging.info('Sampling -- ckpt: %d' % (ckpt,))
            # Wrapper EDM sampling
            processed_mols, MAE1_loss, MAE2_loss = sampling_fn(model, classifier1, classifier2)
            logging.info(f"{config.cond_property1} MAE: %.4f" % MAE1_loss)
            logging.info(f"{config.cond_property2} MAE: %.4f" % MAE2_loss)

            # EDM evaluation metrics
            stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
            logging.info('Number of molecules: %d' % len(sample_rdmols))
            logging.info("3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                         " unique & valid: %.4f, novelty: %.4f" % (stability_res['atom_stable'],
                                                                   stability_res['mol_stable'], rdkit_res['Validity'],
                                                                   rdkit_res['Complete'], rdkit_res['Unique'],
                                                                   rdkit_res['Novelty']))

            # Mose evaluation metrics
            mose_res = mose_metric(sample_rdmols)
            logging.info("3D FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                         mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))

            # 2D evaluation metrics
            stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
            logging.info("2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                         " unique & valid: %.4f, novelty: %.4f" % (stability_res['atom_stable'],
                                                                   stability_res['mol_stable'], rdkit_res['Validity'],
                                                                   rdkit_res['Complete'], rdkit_res['Unique'],
                                                                   rdkit_res['Novelty']))
            mose_res = mose_metric(complete_rdmols)
            logging.info("2D FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                         mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))

            # save sample mols
            if config.eval.save_graph:
                sampler_name = config.sampling.method
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}_{}.pkl".format(ckpt, config.seed))
                with open(graph_file, "wb") as f:
                    pickle.dump(complete_rdmols, f)

            # Substructure Geometry MMD
            if config.eval.sub_geometry:
                sub_geo_mmd_res = sub_geo_mmd_metric(complete_rdmols)
                logging.info("Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
                    sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
                    sub_geo_mmd_res['dihedral_angle_mean']))
                ## bond length
                bond_length_str = ''
                for sym in dataset_info['top_bond_sym']:
                    bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
                logging.info(bond_length_str)
                ## bond angle
                bond_angle_str = ''
                for sym in dataset_info['top_angle_sym']:
                    bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(bond_angle_str)
                ## dihedral angle
                dihedral_angle_str = ''
                for sym in dataset_info['top_dihedral_sym']:
                    dihedral_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(dihedral_angle_str)


run_train_dict = {
    'vpsde': vpsde_train,
    'vpsde_edge': vpsde_edge_train,
    'vpsde_edge_cond': vpsde_edge_cond_train,
    'vpsde_edge_cond_multi': vpsde_edge_cond_multi_train
}


run_eval_dict = {
    'vpsde': vpsde_evaluate,
    'vpsde_edge': vpsde_edge_evaluate,
    'vpsde_edge_cond': vpsde_edge_cond_evaluate,
    'vpsde_edge_cond_multi': vpsde_edge_cond_multi_evaluate
}


def train(config, workdir):
    run_train_dict[config.exp_type](config, workdir)


def evaluate(config, workdir, eval_folder='eval'):
    run_eval_dict[config.exp_type](config, workdir, eval_folder)
