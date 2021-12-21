from typing import Callable, Iterator

import numpy as np

from ._tau import get_tau_series


def event_series_generator(
    tau_series_generator: Callable[..., Iterator[float]],
    *,
    sigma: float = 1e-3,
    hurst: float = 0.5,
    tau_0: float = 0.5,
    boundary: float = 1e-3,
    dt: float = 1,
    n_batch_points: int = 262144,
    n_tau_points: int = -1,
    seed: int = -1,
) -> Iterator[np.ndarray]:
    """Generate event count time series (in physical time space).

    Inter-event time series models are implemented as generators. This
    function creates generator, which serves as a generic intermediary
    extracting event count time series (number of events per unit time
    window) from a selected inter-event time generator.

    Input:
        tau_series_generator:
            Generator to use when generating the series. Use generators
            defined in this module or use their code to build your own
            (e.g., see `markovian_beta` submodule).
        sigma: (default: 1e-3)
            Sigma parameter of the iterative equation driving the
            process. Sigma parameter controls how fast the inter-event
            time values change.
        hurst: (default: 0.5)
            Hurst index of the inter-event time series. If driving noise
            is not fractional Gaussian noise, then this is an effective
            Hurst index (the "faking" process should still generate the
            same stationary inter-event time distribution as the process
            with fGn).
        tau_0: (default: 0.5)
            Initial condition for the inter-event time series.
        boundary: (default: 1e-3)
            Fine-tunes soft boundary conditions the process is allowed
            to approach. Inter-event times confined to interval
            [0+boundary, 1-boundary].
        dt: (default: 1)
            Discretization time step for the event count time series.
        n_batch_points: (default: 262144)
            The desired length of a single batch (sample) of event count
            time series. Note that the length applies to the physical
            time space and not event time space.
        n_tau_points: (default: -1)
            The desired length of a single batch (sample) of the
            inter-event time series. Note that the length applies to the
            event time space (k-space) and not physical time space. It
            is advisable to use numbers which would be powers of 2. New
            sample is generated each time the old runs out when
            generating new event count time series. If negative value is
            passed (which is the default), then the length will be
            approximately `2*n_batch_points` (rounds to nearest power of
            2).
        seed: (default: -1)
            RNG seed. If negative value is passed (which is the
            default), then seed will be randomly generated by
            `np.random.rand(2**20)`.

    Output:
        Generator for sample event count time series.

    Examples:
        ```
        >> from taudiff import event_series_generator, beta_model
        >> esg = event_series_generator(beta_model, sigma=1e-2,
            hurst=0.75, n_batch_points=16, seed=123)
        >> series = next(esg)
        >> print(series)
            [1 3 2 2 2 2 2 2 2 2 2 2 2 3 2 1]
        ```
    """

    def __batch(sigma, hurst, tau_0, boundary, dt, n_tau_points, seed):
        tau_series = get_tau_series(
            tau_series_generator,
            sigma=sigma,
            hurst=hurst,
            tau_0=tau_0,
            boundary=boundary,
            n_points=n_tau_points,
            seed=seed,
        )
        tau_last = tau_series[-1]

        time_series = np.cumsum(tau_series)
        del tau_series

        time_series = np.ceil(time_series / dt).astype(int)
        event_series = np.bincount(time_series)[1:]
        del time_series

        return event_series, tau_last

    if n_tau_points < 0:
        n_tau_points = 2 ** int(np.round(np.log2(n_batch_points) + 1))

    tau_last = tau_0
    event_series = np.array([], dtype=int)
    while True:
        while len(event_series) < n_batch_points:
            if seed >= 0:
                seed = seed + 1
            cur_event_series, tau_last = __batch(
                sigma, hurst, tau_last, boundary, dt, n_tau_points, seed
            )
            event_series = np.concatenate((event_series, cur_event_series))

        yield event_series[:n_batch_points]
        event_series = event_series[n_batch_points:]
