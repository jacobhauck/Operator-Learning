import os
import shutil

import torch
import torch.utils.data
import operatorlearning as ol


def _evaluate_fourier_sine(x, origin, frequencies, amplitudes):
    """
    Evaluate a Fourier Sine series at the given points

    :param x: (..., d_in) points at which to evaluate the series
    :param origin: (d_in,) where the origin of the series is
    :param frequencies: (T, d_in) frequencies of the T terms in the series
    :param amplitudes: (T, d_out) amplitudes of the T terms in the series
    :return: (..., d_out) values of the series at the points
    """
    shape = x.shape[:-1]
    x_shift = x.reshape(-1, x.shape[-1]) - origin[None]  # (n, d_in)
    angles = torch.einsum('nd,Td->nT', x_shift, frequencies)  # (n, T)
    sines = torch.sin(angles)
    vals = torch.einsum('nT,Td->nd', sines, amplitudes)  # (n, d_out)
    return vals.reshape(*shape, -1)  # (..., d_out)


class FourierSineFunction(ol.Function):
    def __init__(self, x, origin, frequencies, amplitudes):
        """
        A function represented by a Fourier sine series

        :param x: (n, d_in) Default sample coordinates (function will be
            interpolated by an oracle, so these don't matter except to define
            the domain when a default is needed)
        :param origin: (d_in,) where the origin of the series is
        :param frequencies: (T, d_in) frequencies of the T terms in the series
        :param amplitudes: (T, d_out) amplitudes of the T terms in the series
        """
        super(FourierSineFunction, self).__init__(
            _evaluate_fourier_sine(x, origin, frequencies, amplitudes), x,
            interpolator=ol.OracleInterpolator(self._evaluate)
        )

        self.register_buffer('origin', origin)
        self.register_buffer('frequencies', frequencies)
        self.register_buffer('amplitudes', amplitudes)

    def _evaluate(self, x):
        return _evaluate_fourier_sine(x, self.origin, self.frequencies, self.amplitudes)

    @staticmethod
    def load(file):
        state_dict = torch.load(file)
        return FourierSineFunction(
            x=state_dict['x'],
            origin=state_dict['origin'],
            frequencies=state_dict['frequencies'],
            amplitudes=state_dict['amplitudes']
        )


class PoissonDataset(torch.utils.data.Dataset):
    def __init__(self, sources, solutions):
        super().__init__()
        self.sources = sources
        self.solutions = solutions
        self.x = None

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError

        if self.x is None:
            return self.sources[item], self.solutions[item]
        else:
            return self.sources[item](self.x), self.solutions[item](self.x)

    @staticmethod
    def load(folder):
        files = os.listdir(folder)
        sources, solutions = [], []
        for file in files:
            if file.startswith('source'):
                sample_id = os.path.splitext(file)[0][len('source'):]
                source_file = os.path.join(folder, file)
                solution_file = os.path.join(folder, 'solution' + sample_id + '.pth')
                sources.append(FourierSineFunction.load(source_file))
                solutions.append(FourierSineFunction.load(solution_file))

        return PoissonDataset(sources, solutions)

    def save(self, folder):
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass
        os.makedirs(folder, exist_ok=True)

        for i, (source, solution) in enumerate(self):
            source_file = os.path.join(folder, 'source' + str(i) + '.pth')
            solution_file = os.path.join(folder, 'solution' + str(i) + '.pth')
            torch.save(source.state_dict(), source_file)
            torch.save(solution.state_dict(), solution_file)


class PoissonDataGenerator(torch.nn.Module):
    def __init__(
            self,
            a, b,
            source_gen,
            device=None
    ):
        """
        Generates source--solution pairs (f, u) of the Poisson equation
            $-\\Delta u = f$
        subject to homogeneous Dirichlet boundary conditions on a rectangular
        domain [a_1, b_1] x [a_2, b_2] x ... x [a_{d_in}, b_{d_in}].

        :param a: (d_in,) tensor of lower bounds of the domain
        :param b: (d_in,) tensor of upper bounds of the domain
        :param source_gen: (int n, rng random) -> (n, T, d_in), (n, T, d_out)
            Module that outputs array of n x T wave number vectors and n x T
            (vector-valued) amplitudes for the T terms in the (sparse) Fourier
            sine series of n random sources
        :param device: PyTorch device on which to do the generation
        """
        super(PoissonDataGenerator, self).__init__()

        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.source_gen = source_gen

        self.to(device)

    def _validate(self, n, wave_numbers, amplitudes):
        """Validates that the source generator output conforms to the convention"""
        wave_numbers_valid = len(wave_numbers.shape) == 3 \
            and wave_numbers.shape[0] == n \
            and wave_numbers.shape[2] == self.a.shape[0]

        amplitudes_valid = len(amplitudes.shape) == 3 \
            and amplitudes.shape[0] == n \
            and amplitudes.shape[1] == wave_numbers.shape[1]

        return wave_numbers_valid and amplitudes_valid

    def forward(self, n):
        """
        Generate n source--solution pairs (f, u)

        :param n: Number of data pairs to generate
        :return: (sources, solutions), two parallel lists of source and
            solution functions
        """
        wave_numbers, amplitudes = self.source_gen(n)
        assert self._validate(n, wave_numbers, amplitudes), \
            (f'Invalid source generator! Wave numbers shape: '
             f'{wave_numbers.shape}, amplitudes shape: {amplitudes.shape}')

        ell = (self.b - self.a)[None, None, :]  # (1 (n), 1 (T), d_in)
        frequencies = (2*torch.pi) * wave_numbers / ell
        rescales = (frequencies ** 2).sum(dim=-1, keepdim=True)
        rescales[rescales == 0] = 1.0
        solution_amplitudes = amplitudes / rescales

        sources, solutions = [], []
        for i in range(n):
            sources.append(FourierSineFunction(
                torch.stack([self.a, self.b]), self.a.clone(),
                frequencies[i], amplitudes[i]
            ))
            solutions.append(FourierSineFunction(
                torch.stack([self.a, self.b]), self.a.clone(),
                frequencies[i], solution_amplitudes[i]
            ))

        return sources, solutions


class DenseSourceGenerator(torch.nn.Module):
    def __init__(self, n_modes, amplitude_decay):
        """
        Generates sources for the Poisson equation by using every combination
        of Fourier modes in each direction, up to the given maximum values and
        sampling the coefficients from a normal distribution with zero mean and
        standard deviation given by a function that depends on the wave number,
        as specified by amplitude_decay.

        :param n_modes: Number of modes to use in each direction (so wave numbers
            from -n_modes to +n_modes will be used). Should be a list of ints
            giving the number of modes to use in each direction.
        :param amplitude_decay: Module (T, d_in) -> (T, d_out), the standard
            deviation of coefficients for the given wave number (this module
            uses uncorrelated output components)
        """
        super(DenseSourceGenerator, self).__init__()

        self.n_modes = n_modes

        wave_numbers_per_dir = [torch.arange(-n, n+1) for n in self.n_modes]
        self.register_buffer(
            'wave_numbers',
            torch.stack(
                torch.meshgrid(wave_numbers_per_dir, indexing='ij'),
                dim=-1
            ).reshape(-1, len(self.n_modes))  # (T, d_in)
        )

        self.register_buffer(
            'amplitude_decays',
            amplitude_decay(self.wave_numbers)  # (T, d_out)
        )

    def forward(self, n):
        """
        Generate wave numbers and amplitudes for n random sources

        :return: (n, T, d_in) wave_numbers, (n, T, d_out) amplitudes
        """
        a_std = torch.randn(
            (n, *self.amplitude_decays.shape),
            dtype=self.amplitude_decays.dtype,
            device=self.amplitude_decays.device
        )  # (n, T, d_out)
        amplitudes = a_std * self.amplitude_decays[None]  # (n, T, d_out)

        wave_numbers = torch.tile(self.wave_numbers, (n, 1, 1))  # (n, T, d_in)

        return wave_numbers, amplitudes
