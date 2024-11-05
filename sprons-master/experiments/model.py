import scipy
import numpy as np
import torch
import torch.nn as nn
import torchdiffeq
import torchdyn.numerics
import datetime


def get_activation_from_name(name):
    if hasattr(torch, name):
        return getattr(torch, name)
    if hasattr(nn.functional, name):
        return getattr(nn.functional, name)
    if name == "identity":
        return lambda x: x
    raise NotImplementedError(name)


class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_hidden_layer: int, nonlinearity: str):
        super(MLP, self).__init__()
        assert hidden_dim > 0
        self.units = [input_dim] + [hidden_dim] * n_hidden_layer + [output_dim]
        self.n_params = sum([self.units[i] * self.units[i + 1] + self.units[i + 1] for i in range(n_hidden_layer + 1)])
        self.act = get_activation_from_name(nonlinearity)
        self.nonlinearity = nonlinearity
        print(f"[DEBUG] MLP initialized: input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}, n_hidden_layer={n_hidden_layer}, nonlinearity={nonlinearity}")

    def segment_params(self, params):
        weights = []
        biases = []
        itr_end = 0
        for i in range(len(self.units) - 1):
            itr_stt = itr_end
            itr_end = itr_stt + self.units[i] * self.units[i + 1]
            weights.append(params[itr_stt:itr_end].view(self.units[i + 1], self.units[i]))
            itr_stt = itr_end
            itr_end = itr_stt + self.units[i + 1]
            biases.append(params[itr_stt:itr_end])
        assert torch.numel(params) == itr_end
        print(f"[DEBUG] Parameters segmented: total_params={torch.numel(params)}, weights_count={len(weights)}, biases_count={len(biases)}")
        return weights, biases

    def init_params(self, params):
        weights, biases = self.segment_params(params)
        try:
            gain = torch.nn.init.calculate_gain(self.nonlinearity)
        except:
            gain = 1.0
        for w in weights:
            nn.init.xavier_normal_(w, gain=gain)
            print(f"[DEBUG] Initialized weight: shape={w.shape}")
        for b in biases:
            nn.init.zeros_(b)
            print(f"[DEBUG] Initialized bias: shape={b.shape}")

    def get_init_params(self):
        theta = torch.zeros(self.n_params)
        self.init_params(theta)
        print(f"[DEBUG] Initial parameters generated: n_params={self.n_params}")
        return theta

    def forward(self, x, params):
        print(f"[DEBUG] Forward pass initiated: input_shape={x.shape}")
        weights, biases = self.segment_params(params)
        with_act = [True] * (len(self.units) - 1) + [False]
        for i, (w, b, a) in enumerate(zip(weights, biases, with_act)):
            x = nn.functional.linear(x, w, b)
            print(f"[DEBUG] Layer {i}: after linear transformation shape={x.shape}")
            if a:
                x = self.act(x)
                print(f"[DEBUG] Layer {i}: after activation shape={x.shape}")
        print(f"[DEBUG] Forward pass completed: output_shape={x.shape}")
        return x


class EDNN(nn.Module):
    def __init__(
        self,
        x_range,
        space_dim: int,
        state_dim: int,
        hidden_dim: int,
        n_hidden_layer: int,
        nonlinearity: str = "tanh",
        sinusoidal: int = 0,
        is_periodic_boundary: bool = True,
        is_zero_boundary: bool = False,
        space_normalization: bool = True,
    ):
        super(EDNN, self).__init__()
        assert not (is_periodic_boundary and is_zero_boundary)
        assert not is_periodic_boundary or space_normalization  # is_periodic_boundary -> space_normalization

        # x_range: [x1,x2,...] x [min, max]
        self.x_range = torch.from_numpy(x_range)
        self.space_normalization = space_normalization
        self.sinusoidal = sinusoidal
        self.is_periodic_boundary = is_periodic_boundary
        if sinusoidal == 0 and is_periodic_boundary:
            self.sinusoidal = 1
        self.is_zero_boundary = is_zero_boundary

        input_dim = space_dim * ((2 * self.sinusoidal) if self.sinusoidal > 0 else 0)
        self.mlp = MLP(input_dim=input_dim, output_dim=state_dim, hidden_dim=hidden_dim, n_hidden_layer=n_hidden_layer, nonlinearity=nonlinearity)

        print(f"[DEBUG] EDNN initialized: space_dim={space_dim}, state_dim={state_dim}, hidden_dim={hidden_dim}, "
              f"n_hidden_layer={n_hidden_layer}, nonlinearity={nonlinearity}, sinusoidal={sinusoidal}, "
              f"is_periodic_boundary={is_periodic_boundary}, is_zero_boundary={is_zero_boundary}, "
              f"space_normalization={space_normalization}")

    def forward(self, x, params):
        # x: batch x [x1,x2,...]
        print(f"[DEBUG] Forward pass started: input_shape={x.shape}, params_shape={params.shape}")

        x_norm = x
        x_range = self.x_range.to(device=x.device, dtype=x.dtype)
        print(f"[DEBUG] x_range after transfer: {x_range}")

        if self.space_normalization:
            x_norm = 2 * (x_norm - x_range[None, :, 0]) / (x_range[None, :, 1] - x_range[None, :, 0]) - 1  # x_norm ∈ [-1,1]
            x_range = 2 * (x_range - x_range[:, 0]) / (x_range[:, 1] - x_range[:, 0]) - 1  # [-1,1]
            print(f"[DEBUG] x_norm after normalization: {x_norm}")

            if self.sinusoidal > 0:
                # x_norm: batch x [sin x1 , sin x2, ..., cos x1, cos x2, ...]
                state_list = []
                for k in range(1, self.sinusoidal + 1):
                    sin_2k_x_norm = torch.sin(torch.pi * 2**(k - 1) * x_norm) * 2**(-(k - 1))
                    cos_2k_x_norm = torch.cos(torch.pi * 2**(k - 1) * x_norm) * 2**(-(k - 1))
                    state_list.append(sin_2k_x_norm)
                    state_list.append(cos_2k_x_norm)
                x_norm = torch.cat(state_list, dim=-1)
                print(f"[DEBUG] x_norm after sinusoidal transformation: {x_norm}")

                x_range = x_range.tile(2 * self.sinusoidal).view(len(x_range), 2, 2 * self.sinusoidal).transpose(0, 1).reshape(len(x_range) * 2 * self.sinusoidal, 2)
                print(f"[DEBUG] x_range after sinusoidal tiling: {x_range}")

        u = self.mlp(x_norm, params)
        print(f"[DEBUG] Output from MLP: u_shape={u.shape}")

        if self.is_zero_boundary:
            x_pos = (x_norm - x_range[None, :, 0]) / (x_range[None, :, 1] - x_range[None, :, 0])
            u = u * 4 * x_pos * (1 - x_pos)  # u=0 at boundary
            print(f"[DEBUG] Output u after applying zero boundary conditions: u_shape={u.shape}")

        print(f"[DEBUG] Forward pass completed: output_shape={u.shape}")
        return u

    def u_t(self, x):
        # Placeholder for u_t method
        pass

    def get_random_sampling_points(self, N):
        x = torch.rand(N, len(self.x_range))
        x = x * (self.x_range[:, 1] - self.x_range[:, 0]) + self.x_range[:, 0]
        print(f"[DEBUG] Random sampling points generated: shape={x.shape}")
        return x

    def get_init_params(self):
        init_params = self.mlp.get_init_params()
        print(f"[DEBUG] Initial parameters obtained: shape={init_params.shape}")
        return init_params


class EDNNTrainer(nn.Module):

    def __init__(self, ednn, log_freq=100, logger=print):
        super(EDNNTrainer, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.get_default_dtype()

        self.ednn = ednn.to(device=self.device)
        self.log_freq = log_freq
        self.nfe = 0
        self.logger = logger
        self.logger(f"[{datetime.datetime.now()}] EDNN, Initialized: The number of parameters is ", self.ednn.mlp.n_params)

    def learn_initial_condition(self, x0, u0, reg: float = 0.0, optim: str = "adam", lr: float = 1e-3, atol: float = 1e-7, max_itr: int = 1000000):
        self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC...")
        assert reg >= 0.0
        x0 = torch.from_numpy(x0).to(device=self.device, dtype=self.dtype)
        u0 = torch.from_numpy(u0).to(device=self.device, dtype=self.dtype)
        
        self.logger(f"[{datetime.datetime.now()}] Initial conditions set: x0 shape {x0.shape}, u0 shape {u0.shape}")

        params = self.ednn.get_init_params().to(device=self.device, dtype=self.dtype)
        params = nn.Parameter(params)

        itr = 0
        optimizer = torch.optim.Adam([params], lr=min(lr, 1e-3), weight_decay=0)

        with torch.set_grad_enabled(True):

            def closure():
                nonlocal itr
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.ednn(x0, params), u0)
                if reg > 0.0:
                    loss = loss + reg * params.__pow__(2).sum()
                if itr % self.log_freq == 0:
                    self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: {itr:*>7}/{max_itr}, Loss: {loss.item():.6e}")
                loss.backward()
                itr += 1
                return loss.item()

            if optim.startswith("adam"):
                self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC by Adam...")
                while itr < max_itr:
                    loss = closure()
                    optimizer.step()
                    self.logger(f"[{datetime.datetime.now()}] Adam Step: itr={itr}, loss={loss:.6e}")
                    if loss < atol:
                        break
            elif optim.startswith("lbfgs"):
                self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC by Adam first...")
                while itr < max_itr // 10:
                    loss = closure()
                    optimizer.step()
                self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC by LBFGS...")
                optimizer = torch.optim.LBFGS([params], lr=lr, max_iter=max_itr - max_itr // 10, tolerance_grad=atol, tolerance_change=0, history_size=100)
                optimizer.step(closure)
            else:
                raise NotImplementedError(self.optim)

        loss = nn.functional.mse_loss(self.ednn(x0, params), u0)
        self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC finished, Loss: {loss.item():.6e}")
        return params.data

    def get_time_derivative(self, equation, params, method, x, reg):
        params = params.flatten()  # remove batch axis added by torchdiffeq
        self.logger(f"[{datetime.datetime.now()}] Getting time derivative... params shape: {params.shape}")

        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            u_t = equation(self.ednn(x, params), x).T
            self.logger(f"[{datetime.datetime.now()}] u_t shape: {u_t.shape}")

        params_to_u = lambda params: self.ednn(x, params)
        J_T = torch.autograd.functional.jacobian(params_to_u, params)
        assert isinstance(J_T, torch.Tensor)
        J = J_T.transpose(0, 1)
        if reg > 0:
            u_dim = u_t.shape[0]
            n_param = params.shape[0]
            v = torch.zeros([u_dim, n_param], device=u_t.device, dtype=u_t.dtype)
            K = np.sqrt(reg / u_dim) * torch.eye(n_param, device=u_t.device, dtype=u_t.dtype).expand(u_dim, -1, -1)

        self.logger(f"[{datetime.datetime.now()}] Jacobian J shape: {J.shape}")

        if method == "collocation":
            if reg > 0:
                u_t = torch.cat([u_t, v], dim=1)
                J = torch.cat([J, K], dim=1)
            ret = torch.linalg.lstsq(J, u_t)
            deriv = ret.solution
        elif method == "gelsd":
            if reg > 0:
                u_t = torch.cat([u_t, v], dim=1)
                J = torch.cat([J, K], dim=1)
            ret = torch.linalg.lstsq(J.cpu(), u_t.cpu(), driver="gelsd")
            deriv = ret.solution.to(device=J.device)
        else:
            M = torch.einsum("dki,dkj->dij", (J, J))
            a = torch.einsum("dki,dk->di", (J, u_t))
            self.logger(f"[{datetime.datetime.now()}] M shape: {M.shape}, a shape: {a.shape}")

            if method == "inversion":
                assert reg == 0.0
                M_inv = torch.linalg.inv(M)
                deriv = torch.einsum("dji,di->dj", (M_inv, a))
            elif method == "optimization":
                if reg > 0:
                    a = torch.cat([a, v], dim=1)
                    M = torch.cat([M, K], dim=1)
                    ret = torch.linalg.lstsq(M, a)
                    deriv = ret.solution
                else:
                    deriv = torch.linalg.solve(M, a)
            else:
                raise NotImplementedError(method)

        self.logger(f"[{datetime.datetime.now()}] Derivative shape: {deriv.shape}")
        return deriv

    def forward(self, t, state):
        if self.nfe % self.log_freq == 0:
            assert t.numel() == 1
            self.logger(f"[{datetime.datetime.now()}] EDNN, Integration: time={t.item():.6e}, nfe={self.nfe}, state mean={state.mean()}, var={state.var()}")
        self.nfe += 1
        equation = self.integration_params["equation"]
        method = self.integration_params["method"]
        x_eval = self.integration_params["x_eval"]
        n_eval = self.integration_params["n_eval"]
        reg = self.integration_params["reg"]

        assert x_eval is None or n_eval == 0
        if x_eval is None:
            x_eval = self.ednn.get_random_sampling_points(n_eval).to(device=self.device, dtype=self.dtype)

        deriv = self.get_time_derivative(equation, state, method=method, x=x_eval, reg=reg)
        return deriv

    def integrate(self, params, equation, method, solver, t_eval, x_eval=None, n_eval: int = 0, reg: float = 0.0, atol: float = 1e-3, rtol: float = 1e-3):
        self.logger(f"[{datetime.datetime.now()}] EDNN, Integration")
        assert t_eval[0] == 0
        assert reg >= 0.0
        t_eval = torch.tensor(t_eval, device=self.device, dtype=self.dtype)
        x_eval = torch.tensor(x_eval, device=self.device, dtype=self.dtype)
        solver = self.solver_replacer(solver)

        self.nfe = 0
        if n_eval > 0:
            self.integration_params = dict(equation=equation, method=method, x_eval=None, n_eval=n_eval, reg=reg)
        else:
            self.integration_params = dict(equation=equation, method=method, x_eval=x_eval, n_eval=0, reg=reg)

        if solver.startswith("dyn"):
            params_evolved = torchdyn.numerics.odeint(self, params.reshape(1, -1), t_eval, solver=solver[3:], atol=atol, rtol=rtol, verbose=True)[1]
        else:
            params_evolved = torchdiffeq.odeint(self, params.reshape(1, -1), t_eval, rtol=rtol, atol=atol, method=solver)
        
        assert isinstance(params_evolved, torch.Tensor)
        params_evolved = params_evolved.squeeze(1)

        self.logger(f"[{datetime.datetime.now()}] Params evolved shape: {params_evolved.shape}")

        u_evolved = torch.stack([self.ednn(x_eval, p) for p in params_evolved], dim=0)
        self.nfe = 0

        self.logger(f"[{datetime.datetime.now()}] u_evolved shape: {u_evolved.shape}")
        return u_evolved, params_evolved

    def solver_replacer(self, solver):
        if solver == "rk23":
            return "bosh3"
        if solver == "rk45":
            return "dopri5"
        return solver
