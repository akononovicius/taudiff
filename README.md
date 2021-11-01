# `taudiff` - diffusion models of inter-event (tau) times

Here we implement diffusive inter-event time models as well as basic time
series conversion tools necessary to study them. 

**Inter-event time models?** Let the (point) process be composed of events
occurring at times `t[k]`. In this context we can define inter-event times
`tau[k]` as time periods between two consecutive events:
`tau[k] = t[k+1] - t[k]`.

**Diffusion?** We allow `tau[k]` to evolve via stochastic iterative
equations (see below), which can be seen as being discrete analog to
stochastic differential equations. In fact in the simplest case `tau[k]`
process is identical to confined Brownian diffusion and it has very
interesting properties.

**Time series conversion tools?** Inter-event time diffusion occurs in the
event time space (or k-space). In the event time space single time tick
corresponds to a single event occurrence. How much time passes in the
physical time space depends on the actual value of the inter-event time.
Conversion tools convert from event time space to physical time space.
Converted series can be further analyzed using standard time series tools.

Lets say in the event time space we have: `tau[0]=0.5`, `tau[1]=0.2`,
`tau[2]=0.2`, `tau[3]=3`. In the physical time space let us count events per
single unit of time, then we would have: `X[0]=3` (events 0, 1 and 2 occur
in between t=0 and t=1), `X[1]=0`, `X[2]=0`, `X[3]=1` (event 3 occurs at 3.9
time units, so we register in the fourth interval).

This module was used to obtain the results and conduct the analysis in the
forthcoming paper (reference will be added later).

## Usage

We are interested in inter-event time distribution, event count distribution
and spectra of event count time series. To obtain these using
`fractional_beta` model you would:

```python
from taudiff import fractional_model, get_tau_series, event_series_generator

# obtain inter-event time series in the k-space (sample 1024 points)
tau_series = get_tau_series(fractional_model, sigma=1e-2, hurst=0.6,
    n_points=1024, seed=123)

# obtain event count time series in the physical time space (sample 2048
# points)
esg = event_series_generator(fractional_model, sigma=1e-2, hurst=0.6,
    n_batch_points=2048, seed=123)
event_series = next(esg)
del esg
```

All implemented models, iterative equations describing the diffusion of
inter-event times, are implemented as generators. Note that these generators
are finite (they return only fixed amount of inter-event times). This
limitation was introduced due to the same limitation present in the
[fractional Gaussian noise generator](https://github.com/akononovicius/fgn-generator-gsl).

`get_tau_series` function helps extract series (as `numpy` array) from the
model (inter-event time generator).

`event_series_generator` function creates infinite generator of event count
time series. In a single iteration it returns event count time series of
length given by `n_batch_points` parameter. If the need arises
`get_tau_series` function might be called multiple times (each time with new
seed) during a single iteration of the event count time series generator. In
general it is advisable to set as big `n_tau_points` as possible to better
preserve long-range memory effects in models such as `fractional_model`.

## Implemented Models

`beta_model` is driven by the following iterative equation:

```
tau[k+1] = tau[k] + sigma^2 * (1/(2*H) - 1) * (1/tau[k] - 1/(1-tau[k])) +
                  + sigma * epsi[k] .
```

The noise, `epsi[k]`, is assumed to be sampled independently from standard
(zero mean and unit variance) Gaussian distribution. Model parameter `H`
allows to reproduce exactly the same stationary distribution as in the
`fractional_model`. `tau[k]` values are bounded within interval `[0, 1]`.

`powerlaw_model` is driven by the following iterative equation:

```
tau[k+1] = tau[k] + sigma^2 * (1/(2*H) - 1) / tau[k] + sigma * epsi[k] .
```

The noise, `epsi[k]`, once again is assumed to be sampled independently from
standard (zero mean and unit variance) Gaussian distribution. Model
parameter `H` allows to reproduce similar stationary distribution as in the
`fractional_model` (distributions are similar for small `tau[k]`). `tau[k]`
values are bounded within interval `[0, 1]`.

Note that with `powerlaw_model` it might not be a good idea to consider
`H<0.5`.

`fractional_model` is driven by the following iterative equation:

```
tau[k+1] = tau[k] + sigma * epsi_H[k] .
```

The noise, `epsi_H[k]`, here is assumed to be a sample of fractional
Gaussian noise. Sampled values have zero mean and unit variance, but
consecutive samples are correlated. `tau[k]` values are bounded within
interval `[0, 1]`.

## Dependencies

* `numpy`
* `ctypes` (part of standard library; if you want to use `fractional_model`)
* `typing` (part of standard library)

For `fractional_model` to work you'll need to compile C source code
(dependency: GNU Scientific Library). If you have relevant dependencies
installed then simply run `make` command in the `fgn/` subdirectory.
This will generated shared object in the `fgn/` subdirectory, where it
should be detected by the responsible submodule.
