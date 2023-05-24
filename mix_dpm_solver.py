# Modify from DPM-Solver repo

import torch
import torch.nn.functional as F
import math
from models.utils import assert_mean_zero_with_mask, sample_center_gravity_zero_gaussian_with_mask


def split_x(x):
    return x[:, :, :3], x[:, :, 3:]

def merge_x(x, y):
    return torch.cat([x ,y], dim=-1)


class DPM_Solver_hybrid:
    def __init__(self, noise_schedule, config):
        """Construct a sampling method based on DPM-solvers for fast 3D geometry graph generation."""
        self.noise_schedule = noise_schedule
        self.cond_x = None
        self.cond_edge_x = None

        self.order = config.sampling.dpm_solver_order
        self.steps = config.sampling.steps
        self.method = config.sampling.dpm_solver_method
        assert config.model.pred_data, "Not support in current version."
        assert config.model.self_cond, "Not support in current version."

    def data_prediction_fn(self, x, t):
        """Return the data prediction model."""
        pass

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling."""
        if skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N+1).to(device)
        else:
            raise ValueError("Unsupported skip_type {}".formate(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """Get the order of sampling by the singlestep DPM-solver."""
        pass

    def ancestral_position_update(self, position_x, position_pred, node_mask, t_start, t_end, last_step=False):
        alpha_t, sigma_t = self.noise_schedule.marginal_prob(t_start)
        alpha_s, sigma_s = self.noise_schedule.marginal_prob(t_end)
        alpha_t_given_s = alpha_t / alpha_s
        sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        position = (alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2) * position_x + \
                        (alpha_s * sigma2_t_given_s / sigma_t ** 2) * position_pred

        if not last_step:
            position = position + sigma * \
                       sample_center_gravity_zero_gaussian_with_mask(position_x.size(), position_x.device, node_mask)

        return position

    def dpm_solver_first_update(self, model_fn, x, node_mask, edge_mask, edge_x, context, t_start, t_end, last_step,
                                pred_start=None, edge_pred_start=None):
        """DPM-Solver-1 (equivalent to DDIM) from time `t_start` to time `t_end`."""
        ns = self.noise_schedule
        bs = x.size(0)
        lambda_start, lambda_end = ns.marginal_lambda(t_start), ns.marginal_lambda(t_end)
        h = lambda_end - lambda_start
        log_alpha_end = ns.marginal_log_mean_coeff(t_end)
        sigma_start, sigma_end = ns.marginal_std(t_start), ns.marginal_std(t_end)
        alpha_end = torch.exp(log_alpha_end)
        phi_1 = torch.expm1(-h)
        position_start, atom_x_start = split_x(x)

        # prediction at t_start
        if pred_start is None and edge_pred_start is None:
            noise_level_start = ns.get_noiseLevel(t_start)
            pred_start, edge_pred_start = model_fn(x, node_mask, edge_mask, edge_x, context,
                                                   torch.ones(bs, device=x.device) * t_start,
                                                   torch.ones(bs, device=x.device) * noise_level_start)

        position_pred_start, atom_pred_start = split_x(pred_start)

        # update data at t_end
        atom_x_end = sigma_end / sigma_start * atom_x_start - alpha_end * phi_1 * atom_pred_start
        edge_x_end = sigma_end / sigma_start * edge_x - alpha_end * phi_1 * edge_pred_start
        position_end = self.ancestral_position_update(position_start, position_pred_start, node_mask,
                                                      t_start, t_end, last_step)
        x_end = merge_x(position_end, atom_x_end)

        return x_end, edge_x_end


    def singlestep_dpm_solver_second_update(self, model_fn, x, node_mask, edge_mask, edge_x, context,
                                            t_start, t_end, last_step, r1=0.5):
        """Singlestep solver DPM-Solver-2 from time `t_start` to time `t_end`. Position using ancestral sampling."""
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        bs = x.size(0)

        # coefficients
        lambda_start, lambda_end = ns.marginal_lambda(t_start), ns.marginal_lambda(t_end)
        h = lambda_end - lambda_start
        lambda_s1 = lambda_start + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_start, log_alpha_s1, log_alpha_end = ns.marginal_log_mean_coeff(t_start), \
                                                       ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t_end)
        sigma_start, sigma_s1, sigma_end = ns.marginal_std(t_start), ns.marginal_std(s1), ns.marginal_std(t_end)
        alpha_s1, alpha_end = torch.exp(log_alpha_s1), torch.exp(log_alpha_end)

        # dpm_solver++
        phi_11 = torch.expm1(-r1 * h)
        phi_1 = torch.expm1(-h)
        position_start, atom_x_start = split_x(x)

        # prediction at t_start
        noise_level_start = ns.get_noiseLevel(t_start)
        pred_start, edge_pred_start = model_fn(x, node_mask, edge_mask, edge_x, context,
                                               torch.ones(bs, device=x.device) * t_start,
                                               torch.ones(bs, device=x.device) * noise_level_start)
        position_pred_start, atom_pred_start = split_x(pred_start)

        # update data at s1
        atom_x_s1 = (sigma_s1 / sigma_start) * atom_x_start - (alpha_s1 * phi_11) * atom_pred_start
        edge_x_s1 = (sigma_s1 / sigma_start) * edge_x - (alpha_s1 * phi_11) * edge_pred_start
        position_s1 = self.ancestral_position_update(position_start, position_pred_start, node_mask, t_start, s1)
        x_s1 = merge_x(position_s1, atom_x_s1)

        # prediction at s1
        noise_level_s1 = ns.get_noiseLevel(s1)
        pred_s1, edge_pred_s1 = model_fn(x_s1, node_mask, edge_mask, edge_x_s1, context,
                                         torch.ones(bs, device=x.device) * s1,
                                         torch.ones(bs, device=x.device) * noise_level_s1)
        position_pred_s1, atom_pred_s1 = split_x(pred_s1)

        # update data at t_end
        atom_x_end = (
            (sigma_end / sigma_start) * atom_x_start
            - (alpha_end * phi_1) * atom_pred_start
            - (0.5 / r1) * (alpha_end * phi_1) * (atom_pred_s1 - atom_pred_start)
        )
        edge_end = (
            (sigma_end / sigma_start) * edge_x
            - (alpha_end * phi_1) * edge_pred_start
            - (0.5 / r1) * (alpha_end * phi_1) * (edge_pred_s1 - edge_pred_start)
        )
        position_end = self.ancestral_position_update(position_s1, position_pred_s1, node_mask,
                                                      s1, t_end, last_step)
        x_end = merge_x(position_end, atom_x_end)
        return x_end, edge_end


    def singlestep_dpm_solver_third_update(self, model_fn, x, node_mask, edge_mask, edge_x, context,
                                           t_start, t_end, last_step, r1=1./3., r2=2./3.):
        """Singlestep solver DPM-Solver-3 from time `s` to time `t`."""
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        bs = x.size(0)

        lambda_start, lambda_end = ns.marginal_lambda(t_start), ns.marginal_lambda(t_end)
        h = lambda_end - lambda_start
        lambda_s1 = lambda_start + r1 * h
        lambda_s2 = lambda_start + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)

        log_alpha_start, log_alpha_s1, log_alpha_s2, log_alpha_end = ns.marginal_log_mean_coeff(t_start), \
            ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t_end)
        sigma_start, sigma_s1, sigma_s2, sigma_end = ns.marginal_std(t_start), ns.marginal_std(s1), \
                                                     ns.marginal_std(s2), ns.marginal_std(t_end)
        alpha_s1, alpha_s2, alpha_end = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_end)

        phi_11 = torch.expm1(-r1 * h)
        phi_12 = torch.expm1(-r2 * h)
        phi_1 = torch.expm1(-h)
        phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
        phi_2 = phi_1 / h + 1.
        # phi_3 = phi_2 / h - 0.5
        position_start, atom_x_start = split_x(x)

        # prediction at t_start
        noise_level_start = ns.get_noiseLevel(t_start)
        pred_start, edge_pred_start = model_fn(x, node_mask, edge_mask, edge_x, context,
                                               torch.ones(bs, device=x.device) * t_start,
                                               torch.ones(bs, device=x.device) * noise_level_start)
        position_pred_start, atom_pred_start = split_x(pred_start)

        # update data at s1
        atom_x_s1 = (sigma_s1 / sigma_start) * atom_x_start - (alpha_s1 * phi_11) * atom_pred_start
        edge_x_s1 = (sigma_s1 / sigma_start) * edge_x - (alpha_s1 * phi_11) * edge_pred_start
        position_s1 = self.ancestral_position_update(position_start, position_pred_start, node_mask, t_start, s1)
        x_s1 = merge_x(position_s1, atom_x_s1)

        # prediction at s1
        noise_level_s1 = ns.get_noiseLevel(s1)
        pred_s1, edge_pred_s1 = model_fn(x_s1, node_mask, edge_mask, edge_x_s1, context,
                                         torch.ones(bs, device=x.device) * s1,
                                         torch.ones(bs, device=x.device) * noise_level_s1)
        position_pred_s1, atom_pred_s1 = split_x(pred_s1)

        # update data at s2
        atom_x_s2 = (sigma_s2 / sigma_start) * atom_x_start - (alpha_s2 * phi_12) * atom_pred_start + \
            r2 / r1 * (alpha_s2 * phi_22) * (atom_pred_s1 - atom_pred_start)
        edge_x_s2 = (sigma_s2 / sigma_start) * edge_x - (alpha_s2 * phi_12) * edge_pred_start + \
            r2 / r1 * (alpha_s2 * phi_22) * (edge_pred_s1 - edge_pred_start)
        position_s2 = self.ancestral_position_update(position_s1, position_pred_s1, node_mask, s1, s2)
        x_s2 = merge_x(position_s2, atom_x_s2)

        # prediction at s2
        noise_level_s2 = ns.get_noiseLevel(s2)
        pred_s2, edge_pred_s2 = model_fn(x_s2, node_mask, edge_mask, edge_x_s2, context,
                                         torch.ones(bs, device=x.device) * s2,
                                         torch.ones(bs, device=x.device) * noise_level_s2)
        position_pred_s2, atom_pred_s2 = split_x(pred_s2)

        # update data at t_end
        atom_x_end = (sigma_end / sigma_start) * atom_x_start - (alpha_end * phi_1) * atom_pred_start + \
                     (1. / r2) * (alpha_end * phi_2) * (atom_pred_s2 - atom_pred_start)
        edge_x_end = (sigma_end / sigma_start) * edge_x - (alpha_end * phi_1) * edge_pred_start + \
                     (1. / r2) * (alpha_end * phi_2) * (edge_pred_s2 - edge_pred_start)
        position_end = self.ancestral_position_update(position_s2, position_pred_s2, node_mask, s2, t_end, last_step)

        x_end = merge_x(position_end, atom_x_end)
        return x_end, edge_x_end


    def multistep_dpm_solver_second_update(self, model_fn, x, node_mask, edge_mask, edge_x, context,
                                           model_prev_list, t_prev_list, t, last_step):
        """Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`."""
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        pred_prev_1, edge_pred_prev_1 = model_prev_1
        pred_prev_0, edge_pred_prev_0 = model_prev_0
        _, atom_pred_prev_1 = split_x(pred_prev_1)
        position_pred_prev_0, atom_pred_prev_0 = split_x(pred_prev_0)
        position_prev_0, atom_x_prev_0 = split_x(x)

        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1 , lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), \
                                                  ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        phi_1 = torch.expm1(-h)

        # update data at time t
        D1_0_atom = (1. / r0) * (atom_pred_prev_0 - atom_pred_prev_1)
        D1_0_edge = (1. / r0) * (edge_pred_prev_0 - edge_pred_prev_1)
        atom_x_t = (sigma_t / sigma_prev_0) * atom_x_prev_0 \
                   - (alpha_t * phi_1) * atom_pred_prev_0 \
                   - 0.5 * (alpha_t * phi_1) * D1_0_atom
        edge_x_t = (sigma_t / sigma_prev_0) * edge_x \
                   - (alpha_t * phi_1) * edge_pred_prev_0 \
                   - 0.5 * (alpha_t * phi_1) * D1_0_edge
        position_t = self.ancestral_position_update(position_prev_0, position_pred_prev_0, node_mask,
                                                    t_prev_list[-1], t, last_step)
        x_t = merge_x(position_t, atom_x_t)
        return x_t, edge_x_t


    def singlestep_dpm_solver_update(self, model_fn, x, node_mask, edge_mask, edge_x, context, t_start, t_end,
                                     last_step, order, r1=None, r2=None):
        """Singlestep DPM-solver++ with the order `order` from time `s` to time `t`."""
        if order == 1:
            return self.dpm_solver_first_update(model_fn, x, node_mask, edge_mask, edge_x, context,
                                                t_start, t_end, last_step)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(model_fn, x, node_mask, edge_mask, edge_x, context,
                                                            t_start, t_end, last_step, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(model_fn, x, node_mask, edge_mask, edge_x, context,
                                                           t_start, t_end, last_step, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order Error")

    def multistep_dpm_solver_update(self, model_fn, x, node_mask, edge_mask, edge_x, context, model_prev_list,
                                    t_prev_list, t, last_step, order):
        if order == 1:
            return self.dpm_solver_first_update(model_fn, x, node_mask, edge_mask, edge_x, context,
                                                t_prev_list[-1], t, last_step,
                                                pred_start=model_prev_list[-1][0],
                                                edge_pred_start=model_prev_list[-1][1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(model_fn, x, node_mask, edge_mask, edge_x,
                                                           context, model_prev_list, t_prev_list, t, last_step)
        else:
            raise ValueError("Solver order Error")

    def get_model_fn(self, model):
        def model_fn(x, node_mask, edge_mask, edge_x, context, vec_t, noise_level):
            pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                        cond_x=self.cond_x, cond_edge_x=self.cond_edge_x, context=context)
            self.cond_x, self.cond_edge_x = pred_t, edge_pred_t
            return pred_t, edge_pred_t
        return model_fn

    @torch.no_grad()
    def sampling(self, model, x, node_mask, edge_mask, edge_x, context=None,
                 t_start=None, t_end=None, skip_type='time_uniform'):
        """Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`."""

        steps = self.steps
        order = self.order
        self.cond_x, self.cond_edge_x = None, None
        model_fn = self.get_model_fn(model)

        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start

        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be " \
                                    "in [1 / N, 1], where N is the length of betas array"

        device = x.device
        if self.method == 'singlestep_fixed':
            K = steps // order
            orders = [order,] * K
            timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
            for step, order in enumerate(orders):
                t_start, t_end = timesteps_outer[step], timesteps_outer[step + 1]  # t_i, t_{i-1}
                timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=t_start.item(), t_0=t_end.item(),
                                                      N=order, device=device)
                lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                h = lambda_inner[-1] - lambda_inner[0]
                r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                x, edge_x = self.singlestep_dpm_solver_update(model_fn, x, node_mask, edge_mask, edge_x, context,
                                t_start, t_end, step==len(orders)-1, order=self.order, r1=r1, r2=r2)

        elif self.method == 'multistep':
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            # Init the initial values.
            step = 0
            t = timesteps[step]
            t_prev_list = [t]
            model_prev_list = [model_fn(x, node_mask, edge_mask, edge_x, context,
                torch.ones(x.size(0), device=x.device) * t,
                torch.ones(x.size(0), device=x.device) * self.noise_schedule.get_noiseLevel(t))]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for step in range(1, order):
                t = timesteps[step]
                x, edge_x = self.multistep_dpm_solver_update(model_fn, x, node_mask, edge_mask, edge_x, context,
                                                             model_prev_list, t_prev_list, t,
                                                             last_step=False, order=step)
                t_prev_list.append(t)
                model_prev_list.append(model_fn(x, node_mask, edge_mask, edge_x, context,
                    torch.ones(x.size(0), device=x.device) * t,
                    torch.ones(x.size(0), device=x.device) * self.noise_schedule.get_noiseLevel(t)))
            # Compute the remaining values by `order`-th order multistep DPM-Solver.
            for step in range(order, steps + 1):
                t = timesteps[step]
                step_order = order
                x, edge_x = self.multistep_dpm_solver_update(model_fn, x, node_mask, edge_mask, edge_x, context,
                                                             model_prev_list, t_prev_list, t,
                                                             last_step=step==steps, order=step_order)
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = t
                # We do not need to evaluate the final model value.
                if step < steps:
                    model_prev_list[-1] = model_fn(x, node_mask, edge_mask, edge_x, context,
                        torch.ones(x.size(0), device=x.device) * t,
                        torch.ones(x.size(0), device=x.device) * self.noise_schedule.get_noiseLevel(t))
        else:
            raise ValueError("Get wrong method {}".format(self.method))

        assert_mean_zero_with_mask(x[:, :, :3], node_mask)
        return x, edge_x
