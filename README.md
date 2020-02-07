## Control-Gym
The control-gym module contains the code for the Ball on Plate (BoP) and Mass-Spring-Damper (MSD) environments. Both environments are implemented as [OpenAI Gym](https://gym.openai.com/) environments ([GitHub](https://github.com/openai/gym)). Therefore, their usage is similar to other gym environments.

Usage:
```python
    import control-gym as cgym
    env = cgym.make("model name", "integrator name", **kwargs)
```

The model string defines which model is loaded, the integrator string defines which method is used for integration (default is forward euler), and the key word arguments specify the arguments passed to the specific model constructor (see documentation for further info).

The model names registered are:
1. "MassSpringDamper-v0": Standard MSD model, use N=xyz to specify the number of masses (agents).
2. "BoPSimple1D-v0": Standard two dimensional BoP model, i.e. ball on a bar, with simplified dynamics (dynamics are in one dimension).
3. "BoPUncertain1D-v0": Standard two dimensional BoP model, i.e. ball on a bar, but with added uncertain dynamics.

The integrator names registered are:
1. "Euler": Forward euler integration.
2. "ImplicitEuler": Backward euler integration.
3. "Heun": Heun's method (two-stage, second-order Runge-Kutta method).
4. "RK45": Explicit, fifth(4)-order Runge-Kutta.

Additionally, to the standard gym interface, the environments implemented in the models module have a "simulate" method. This method simulates the model over a given time and for a given input (further detail in documentation).

