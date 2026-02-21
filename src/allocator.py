class GreenAllocator:
    def __init__(self, approaches, C, alpha_up=0.8, alpha_down=0.25,
                 beta=0.6, T_base=90, G_min=10, G_max=60,
                 W_th=45, delta=0.15, T_alloc=10):
        self.approaches = approaches
        self.C = C
        self.alpha_up, self.alpha_down = alpha_up, alpha_down
        self.beta, self.T_base = beta, T_base
        self.G_min, self.G_max = G_min, G_max
        self.W_th, self.delta, self.T_alloc = W_th, delta, T_alloc
        self.prev_EVU = {a: 0.0 for a in approaches}

    def step(self, EVU_raw, queue, emergency):
        EVU_smooth = {}
        for a in self.approaches:
            if EVU_raw[a] >= self.prev_EVU[a]:
                alpha = self.alpha_up
            else:
                alpha = self.alpha_down
            EVU_smooth[a] = alpha * EVU_raw[a] + (1 - alpha) * self.prev_EVU[a]
            self.prev_EVU[a] = EVU_smooth[a]

        D = {a: EVU_smooth[a] / max(1e-6, self.C[a]) for a in self.approaches}
        Wmax = max(1e-6, max(queue.values()))
        W_norm = {a: queue[a]/Wmax for a in self.approaches}
        Dprime = {a: D[a] + self.beta * W_norm[a] for a in self.approaches}

        G = {a: max(self.G_min, min(self.G_max, Dprime[a] * self.T_base)) for a in self.approaches}
        return EVU_smooth, Dprime, G
