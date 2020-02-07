import numpy as np
from scipy.integrate import solve_ivp as odesolver
import logging
import sys
import gym
from gym import spaces, logger
from gym.utils import seeding
import functools
from scipy.integrate import RK45
from collections import namedtuple

""" Logger Setup """
# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create STDERR handler
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
ch.setFormatter(formatter)

# Set STDERR handler as the only handler
logger.handlers = [ch]

# prevent propagation
logger.propagate = False


class Integrator(object):
    """Integrator class

        This class defines various integrators to use with any continuous
        time system, in order to discretize it. The following methods are
        implemented:
            - forward euler
            - backward euler
            - Heun
            - RK45

        The integrator object will use the continuous time equation of
        motions and compute the next discrete state update.

        Example:
            integrator = Integrator("integrator name")

        Note:
            Newly added integrator methods need to be registered in the
            INTEGRATORS dictionary.
    """
    def __init__(self, method='default'):
        """Initialize Integrator class.

            Args:
                method (str): Name of integrator method, as registred in the
                    INTEGRATORS dictionary.

            Raises:
                AssertionError: If no valid key for the INTEGRATORS dictionary
                    is passed to the method.
        """
        assert method in INTEGRATORS, "No valid integrator key! Valid keys are {}".format([m for m in INTEGRATORS.keys()])
        self._method = getattr(self, INTEGRATORS[method])

    def __call__(self, EoM, state, action, dt):
        """Call Integrator class.
        
            This method calls the previously defined integration method.
            Usage: Integrator(EoM, state, action, dt)

            Args:
                EoM (fun): Continuous time equations of motion implemented
                    as a function. State and action will be passed to this
                    function.
                state (np.array): Numpy array of the current state. Has to
                    be compatible with the EoM function!
                action (np.array): Numpy array of the current action. Has to
                    be compatible with the EoM function!
                dt (float): Time discretization.

            Returns:
                state (np.array): Numpy array of the next state. Computed using
                    the continuous time equations of motions, discretized with
                    the chosen method.
        """
        return self._method(EoM, state, action, dt)

    def _euler(self, EoM, state, action, dt):
        """Forward Euler method.
        
            This method implements the forward euler integration method.

            Args:
                As in __call__.
        """
        update = EoM(state, action)
        return state + dt*update

    def _implicit_euler(self, EoM, state, action, dt):
        """Backward Euler method.
        
            This method implements the backward euler integration method.

            Args:
                As in __call__.

            Note:
                The implemented version is actually semi-implicit.
        """
        update = EoM(state, action)
        explicit_step = np.zeros(state.shape)
        implicit_step = np.zeros(state.shape)
        explicit_step[1::2] = state[1::2] + dt * update[1::2]
        implicit_step[0::2] = state[0::2] + dt * explicit_step[1::2]
        return explicit_step + implicit_step

    def _heun(self, EoM, state, action, dt):
        """Heun method.
        
            This method implements the Heun's integration method.

            Args:
                As in __call__.
        """
        update = EoM(state, action)
        euler_step = state + dt * update
        euler_update = EoM(euler_step, action)
        return state + dt/2 * (update + euler_update)

    def _RK45(self, EoM, state, action, dt):
        """Runge-Kutta 5(4)
        
            This method implements the RK45 integration method.
            It uses the implementation provided by scipy.

            Args:
                As in __call__.
        """
        def model(t, y):
            return EoM(y, action)
        solver =  RK45(model, 0, state, dt)
        while solver.status != 'finished':
            solver.step()
        return solver.y

""" Registered integration methods """
INTEGRATORS = {'default': '_euler',
               'Euler': '_euler',
               'ImplicitEuler': '_implicit_euler',
               'Heun': '_heun',
               'RK45': '_RK45'}

def make(model='default', integration_method='default', **kwargs):
    """Initialize model object.

        This method initializes a chosen model with the chosen integration
        method and allows further arguments to be passed to the model
        constructor. The interface follows the same guidelines as the OpenAI
        gym module, in order to simplify the integration. The following models
        are implemented:
            - BoPSimple1D-v0
            - BoPUncertain1D-v0
            - BoPFull1D-v0 (only EoM)
            - BoPFull2D-v0 (only EoM)
            - MABoP1D-v0 (experimental)
            - MassSpringDamper-v0

        Args:
            model (str): Name of model, as registred in the MODELS dictionary.
            integration_method (str): Name of integrator method, as registred
                in the INTEGRATORS dictionary.
            **kwargs (dict): Additional arguments, which are passed to the
                model constructor.

        Raises:
            AssertionError: If no valid key for the MODELS or INTEGRATORS dictionary
                is passed to the method.
            Error/Warning: Any specific error or warning raised by a models
                constructor

        Returns:
            ModelInstance (obj): Instance of the chosen model, intitialized with the
                given arguments.
    """
    assert model in MODELS, "No valid model key! Valid keys are {}".format([m for m in MODELS.keys()])
    return MODELS[model](integration_method, **kwargs)

class BallOnPlate(gym.Env):
    """Ball on Plate class.

        This class acts as a parent class to all specific Ball on Plate classes
        implemented below. It implements the methods shared by all BoP classes
        and defines the basic interface.
    """
    def __init__(self,
                 integration_method='default',
                 dt=0.01,           #s
                 ball_mass=0.0014,  #kg
                 ball_radius=0.005, #m
                 plate_mass=0.27,   #kg
                 plate_length=1.0,  #m
                 plate_width=1.0    #m
                ):
        """Initialize Ball on Plate class.
        
            This method initalizes all physical quantities and builds the
            integration method.

            Args:
                integration_method (str): Name of integrator method, as registred
                    in the INTEGRATORS dictionary.
                dt (float): Time discretization in seconds.
                ball_mass (float): Mass of ball in kilograms.
                ball_radius (float): Radius of ball in meters.
                plate_mass (float): Mass of plate in kilograms.
                plate_length (float): Length of plate in meters.
                plate_width (float): Width of plate in meters.
        """
        self._gravity = 9.81
        self._ball_mass = ball_mass
        self._ball_radius = ball_radius
        self._plate_mass = plate_mass
        self._plate_length = plate_length
        self._plate_width = plate_width

        # change inertias directly if needed
        self._ball_inertia = 2/5 * self._ball_mass * self._ball_radius**2
        self._plate_inertia = 1/3 * self._plate_mass * self._plate_length**2
        
        self._M = self._ball_mass + self._ball_inertia/self._ball_radius**2
        self._I = self._ball_inertia + self._plate_inertia

        self._dt = dt

        self._input = None
        self._done = False
        self.viewer = None
        self._integrate = Integrator(integration_method)

        self.seed()
     
    def _EoM(self, state, action):
        """Equations of motion.
        
            This method implements the continuous time equations of motion of
            the Ball on Plate model. This method needs to be implemented for
            each specific BoP model individually.

            Args:
                state (np.array): Numpy array of the current state.
                action (np.array): Numpy array of the current action.

            Raises:
                NotImplementedError: If the generic parent BoP class is called.
        """
        raise NotImplementedError

    def _model(self, t, y):
        """Model wrapper for simulation.
        
            This method implements a wrapper for the simulate method. The solve_ivp
            method of scipy doesn't allow the use of an input. This wrapper adds this
            functionality, by querying the _input method in each time step to get
            the current action value.

            Args:
                t (np.array): Numpy array of the current time.
                y (np.array): Numpy array of the current state.

            Note:
                The naming is adapted to the scipy.integrate module.
        """
        action = self._input(t)
        return self._EoM(y, action)

    def simulate(self, t_span, init_state, u, solver='RK45', dense_output=True):
        """Method to simulate the model forward in time.
        
            This method allows to simulate the model forward in time, given
            an input. The state of the model is not affected, hence calling
            simulate in between calls to step, has no effect!

            The interface is adapted from the solve_ivp method of the
            scipy.integrate module. For more information have a look at the
            scipy documentation.

            Args:
                t_span (2-tuple of floats): Integration interval (t0, tf).
                init_state (np.array): Numpy array of the initial state.
                u (fun): Callable function, returning the input of the current
                    time step. Should accept time as an argument.
                solver (str): Wether to use the internal integrator ("internal")
                    or to use the specified solver from scipy.integrate.
                dense_output (bool): Wether to return the complete time vector
                    or use a sparse representation.
        """
        if solver == 'internal':
            Iteration = namedtuple('Iteration', ('t', 'y', 'u'))
            output = []
            state = init_state
            t = t_span[0]
            output.append(Iteration(t=t, y=state, u=None))
            for i in range(int(t_span[1]//self._dt)):
                action = u(t)
                state = self._integrate(self._EoM, state, action, self._dt)
                t += self._dt
                output.append(Iteration(t=t, y=state, u=action))
            output = Iteration(*zip(*output))
            return output
        else:
            self._input = u
            return odesolver(self._model, t_span, init_state, method=solver, dense_output=dense_output)

    def step(self, action):
        """Environment step.
        
            This method implements the standard gym environment step method.
            The method should compute one step of the model, given the current
            state stored in the class and the provided action. See gym documentation
            for further information.

            This method needs to be implemented for each specific BoP model
            individually.

            Args:
                action (np.array): Numpy array of the action to apply to the
                    environment.

            Returns:
                state (np.array): Numpy array of the next state.
                reward (float): Reward for the step taken.
                done (bool): True, if the episode is terminated. False, otherwise.
                info (dict): Optional info dictionary.

            Raises:
                NotImplementedError: If the generic parent BoP class is called.
        """
        raise NotImplementedError

    def _reset(self):
        """Environment reset.
        
            This method resets the environment. This method should be called after
            an episode is finished. This is a standard method of all OpenAI gym
            environments, see gym documentation for further information.

            Here only the attributes common to all BoP models are reset.
            
            Args:
                See specifc methods.

            Returns:
                See specifc methods.
        """
        self._t = 0
        self._input = None
        self._done = False

    def seed(self, seed=None):
        """Environment seed.
        
            This method sets the random seed of the environment. This is a standard
            method of all OpenAI gym environments, see gym documentation for further
            information.

            Args:
                seed (int): Random seed.

            Returns:
                seed (int): Random seed.

            Note:
                This sets the random generator instance as an attribute of the class.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Close environment renderer
        
            This method closes the environment renderer. This is a standard
            method of all OpenAI gym environments, see gym documentation for further
            information.

            Args:
                None.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        """Environment renderer.
        
            This method implements a renderer of the environment, in order to allow for
            a visual output to the screen. This is a standard method of all OpenAI gym
            environments, see gym documentation for further information.

            This method needs to be implemented for each specific BoP model
            individually.

            Args:
                mode (str): Render mode.

            Raises:
                NotImplementedError: If the generic parent BoP class is called.
        """
        raise NotImplementedError

    """INFO"""
    def info(self):
        """Environment info.
        
            This method prints the info and settings of the model to the logging stream.

            Args:
                None.
        """
        logger.info("MODEL INFO")
        logger.info(self.__class__)
        if self.__doc__ is None:
            logger.info("No documentation found.")
        else:
            logger.info(self.__doc__)
        logger.info("INTEGRATOR INFO")
        logger.info(self._integrate._method.__name__)
        if self._integrate._method.__doc__ is None:
            logger.info("No documentation found.")
        else:
            logger.info(self._integrate._method.__doc__)
        return

    def get_state(self):
        """Print current state.
        
            This method prints the current time and state to the logging stream.

            Args:
                None.
        """
        logger.info("Time {0}".format(self._t))
        logger.info("State {0}".format(self._state))
        return {"time": self._t, "state": self._state}

class Simplest1D(BallOnPlate):
    """Simplified 2D Ball on Plate class.

        This class implements the simplified 2D model. The 2D model corresponds
        to a Ball on a Bar model. The simplified model, assumes the mass of the
        plate to be much larger than the mass of the ball, which results in some
        key simplifications. This simplifications render the model linear and akin
        to the double integrator.
        See the report for further details on the simplifications and the derivation.
    """
    def __init__(self, integration_method, initial_state=None, **kwargs):
        super(Simplest1D, self).__init__(integration_method, **kwargs)
        self._x_max = self._plate_length 
        self._x_min = 0.0
        self._v_max = np.finfo(np.float32).max
        self._v_min = -np.finfo(np.float32).max

        self._low = np.array([self._x_min, self._v_min])
        self._high = np.array([self._x_max, self._v_max])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self._low, self._high, dtype=np.float32)

        self.reset(initial_state)

    def _EoM(self, state, action):
        """Equations of motion for simplified 2D Ball on Plate model.

            See generic method of parent class for more information.

            Args:
                state (np.array): Numpy array of the current state.
                action (np.array): Numpy array of the current action.
        """
        alpha = 0.5
        if action == 0:
            u = -alpha
        elif action == 1:
            u = 0
        elif action == 2:
            u = alpha
        else:
            assert np.abs(action) <= alpha
            u = action
        x = state[0]
        xd = state[1]

        #EoM
        xdd = -self._gravity*self._ball_mass/self._M * u/self._plate_length
        return np.array([xd, xdd])

    def reset(self, initial_state=None):
        """Reset method for simplified 2D Ball on Plate model.
        
            See generic method of parent class for more information.

            Args:
                initial_state (np.array): Numpy array of the state, which
                    should be set as the inital state. If None, the position
                    is chosen randomly, while the velocity is always zero.

            Returns:
                state (np.array): Numpy array containing the inital state.
        """
        super(Simplest1D, self)._reset()
        self._input = 1
        if initial_state is None:
            state = self.observation_space.sample()
            self._state = np.array([state[0], 0.0])
        else:
            self._state = initial_state
        return self._state

    def _reward(self, done, state):
        """Reward method for simplified 2D Ball on Plate model.
        
            This method returns the reward given the state and if the episode
            is finished or not. A reward of 1000 is given if the episode was
            successful, i.e. the episode ended with the ball standing still in
            the middle of the plate. If the ball falls off the plate, the reward
            is -1000 and if the episode finishes somewhere on the plate but not
            the middle the reward is -100. Additionally, in each time step an
            reward of 1 is given if the ball is moving towards the middle of the
            plate and -1 if it moves in the opposite direction.

            Args:
                done (bool): True if the episode is finished, False otherwise.
                state (np.array): Numpy array containing the current state.

            Returns:
                reward (float): Reward of the current time step.
        """
        x,v = state
        if not done:
            reward = -np.sign(x-0.5)*np.sign(v)
        else:
            if 0.498 <= x <= 0.502 and np.abs(v) < 0.01:
                reward = 1000
            elif self._t > 10:
                reward = -100
            else:
                reward = -1000
        return reward

    def step(self, action):
        """Step method of the simplified 2D Ball on Plate model.
        
            This method implements the standard gym environment step method.
            The method computes the next state and reward, given the action.
            The episode is terminated if the ball fall of the plate, the ball
            stops in the middle of the plate or if the time exceeds 10 seconds.
            For more information about the reward see the reward method.

            See generic method of parent class for more information.

            Args:
                action (np.array): Numpy array of the action to apply to the
                    environment.

            Returns:
                state (np.array): Numpy array of the next state.
                reward (float): Reward for the step taken.
                done (bool): True, if the episode is terminated. False, otherwise.
                info (dict): Optional info dictionary.
        """
        if not self._done:
            state = self._state
            dt = self._dt
            t = self._t
            self._input = action
            self._state = self._integrate(self._EoM, state, action, dt)
            self._t += self._dt

            x, v = self._state

            done = not self.observation_space.contains(self._state) \
                    or self._t > 10 \
                    or ( 0.498 <= x <= 0.502 and np.abs(v) < 0.01)
            self._done = bool(done)
        else:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        reward = self._reward(self._done, self._state)
        return self._state, reward, self._done, {}

    def render(self, mode='human'):
        """Render method for simplified 2D Ball on Plate model.

            See generic method of parent class for more information.

            Args:
                mode (str): Render mode.
        """
        screen_width = 600
        screen_height = 400

        x_threshold = 2
        world_width = x_threshold*2
        scale = screen_width/world_width
        polewidth = 10
        polelen = scale * (self._plate_length)
        ballrad = 10*scale * self._ball_radius

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -polewidth/2,polelen-polewidth/2,polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(screen_width/2-polelen/2, screen_height/2))
            pole.add_attr(self.poletrans)
            self.viewer.add_geom(pole)
            self.ball = rendering.make_circle(ballrad)
            self.balltrans = rendering.Transform(translation=(screen_width/2-polelen/2, screen_height/2+polewidth/2+ballrad))
            self.ball.add_attr(self.balltrans)
            self.ball.set_color(.5,.5,.8)
            self.viewer.add_geom(self.ball)

            self._pole_geom = pole

        if self._state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polelen-polewidth/2,polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x,xd = self._state
        action = self._input
        alpha = 0.5
        if action == 0:
            a = -alpha
        elif action == 1:
            a = 0
        else:
            a = alpha

        ballx = x*scale * np.cos(a) + screen_width/2-polelen/2
        bally = x*scale * np.sin(a) + screen_height/2+polewidth/2+ballrad
        self.balltrans.set_translation(ballx, bally)
        self.poletrans.set_rotation(a)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class Uncertain1D(BallOnPlate):
    """Simplified 2D Ball on Plate with uncertainty class.

        This class implements the simplified 2D model, given some uncertain
        dynamics. The uncertainty stems from the additional dynamics of the
        change in input. This allows to model the Ball on Plate system with
        the nominal linear dynamics and run it on the slightly non-linear
        environment, implemented in this class.

        Note:
            Taking into account, that the invariance-base safety framework
            assumes the model dynamics to be centered around the origin, the
            dynamics of this model are centered around the origin, too.
            Contrary, to the standard model where this is not the case!
    """
    def __init__(self, integration_method, initial_state=None, **kwargs):
        super(Uncertain1D, self).__init__(integration_method, **kwargs)
        self._x_max = self._plate_length / 2
        self._x_min = - self._plate_length / 2
        self._v_max = np.finfo(np.float32).max
        self._v_min = -np.finfo(np.float32).max
        self._u_max = 0.5
        self._u_min = -0.5

        self._low = np.array([self._x_min, self._v_min])
        self._high = np.array([self._x_max, self._v_max])

        self.action_space = spaces.Box(np.array([self._u_min]), np.array([self._u_max]), dtype=np.float32)
        self.observation_space = spaces.Box(self._low, self._high, dtype=np.float32)

        self.reset(initial_state)

    def _EoM(self, state, action):
        """Equations of motion for simplified 2D Ball on Plate model
            with uncertainty.
            
            The difference between the standard 2D model and the one with
            uncertainty, lies in the dynamics. The model with uncertainty
            incorperates the dynamics on the input into the equations of
            motion. Therefore, the environment is non-linear compared to
            the linear model of the standard version. This allows to consider
            the nominal linear dynamcis of the standard model in a safety
            framework and run it on this non-linear model.

            See generic method of parent class for more information.

            Args:
                state (np.array): Numpy array of the current state.
                action (np.array): Numpy array of the current action.
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        x = state[0]
        xd = state[1]
        u_old = self._input
        u = action.item()

        #EoM
        xdd = self._ball_mass/self._M *((u - u_old)**2/(self._plate_length**2 - u**2) * x - self._gravity * u/self._plate_length)
        return np.array([xd, xdd])

    def reset(self, initial_state=None):
        """Reset method for simplified 2D Ball on Plate model with uncertainty.
        
            See generic method of parent class for more information.

            Args:
                initial_state (np.array): Numpy array of the state, which
                    should be set as the inital state. If None, the position
                    is chosen randomly, while the velocity is always zero.

            Returns:
                state (np.array): Numpy array containing the inital state.
        """
        super(Uncertain1D, self)._reset()
        self._input = 1
        if initial_state is None:
            state = self.observation_space.sample()
            self._state = np.array([state[0], 0.0])
        else:
            self._state = initial_state
        return self._state

    def _reward(self, done, state):
        """Reward method for simplified 2D Ball on Plate model with uncertainty.
        
            This method returns the reward given the state and if the episode
            is finished or not. A reward of 1000 is given if the episode was
            successful, i.e. the episode ended with the ball standing still in
            the middle of the plate. If the ball falls off the plate, the reward
            is -1000 and if the episode finishes somewhere on the plate but not
            the middle the reward is -100. Additionally, in each time step an
            reward of 1 is given if the ball is moving towards the middle of the
            plate and -1 if it moves in the opposite direction.

            Args:
                done (bool): True if the episode is finished, False otherwise.
                state (np.array): Numpy array containing the current state.

            Returns:
                reward (float): Reward of the current time step.
        """
        x,v = state
        if not done:
            reward = -np.sign(x)*np.sign(v)
        else:
            if -0.002 <= x <= 0.002 and np.abs(v) < 0.01:
                reward = 1000
            elif self._t > 5:
                reward = -100
            else:
                reward = -1000
        return reward

    def step(self, action):
        """Step method of the simplified 2D Ball on Plate model with uncertainty.
        
            This method implements the standard gym environment step method.
            The method computes the next state and reward, given the action.
            The episode is terminated if the ball fall of the plate, the ball
            stops in the middle of the plate or if the time exceeds 5 seconds.
            For more information about the reward see the reward method.

            See generic method of parent class for more information.

            Args:
                action (np.array): Numpy array of the action to apply to the
                    environment.

            Returns:
                state (np.array): Numpy array of the next state.
                reward (float): Reward for the step taken.
                done (bool): True, if the episode is terminated. False, otherwise.
                info (dict): Optional info dictionary.
        """
        if not self._done:
            state = self._state
            dt = self._dt
            t = self._t
            self._state = self._integrate(self._EoM, state, action, dt)
            self._input = action.item()
            self._t += self._dt

            x, v = self._state

            done = not self.observation_space.contains(self._state) \
                    or self._t > 5 \
                    or ( -0.002 <= x <= 0.002 and np.abs(v) < 0.01)
            self._done = bool(done)
        else:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        reward = self._reward(self._done, self._state)
        return self._state, reward, self._done, {}

    def render(self, mode='human'):
        """Render method for simplified 2D Ball on Plate model with uncertainty.

            See generic method of parent class for more information.

            Args:
                mode (str): Render mode.
        """
        raise NotImplementedError
        screen_width = 600
        screen_height = 400

        x_threshold = 2
        world_width = x_threshold*2
        scale = screen_width/world_width
        polewidth = 10
        polelen = scale * (self._plate_length)
        ballrad = 10*scale * self._ball_radius

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -polewidth/2,polelen-polewidth/2,polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(screen_width/2-polelen/2, screen_height/2))
            pole.add_attr(self.poletrans)
            self.viewer.add_geom(pole)
            self.ball = rendering.make_circle(ballrad)
            self.balltrans = rendering.Transform(translation=(screen_width/2-polelen/2, screen_height/2+polewidth/2+ballrad))
            self.ball.add_attr(self.balltrans)
            self.ball.set_color(.5,.5,.8)
            self.viewer.add_geom(self.ball)

            self._pole_geom = pole

        if self._state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polelen-polewidth/2,polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x,xd = self._state
        action = self._input
        alpha = 0.5
        if action == 0:
            a = -alpha
        elif action == 1:
            a = 0
        else:
            a = alpha

        ballx = x*scale * np.cos(a) + screen_width/2-polelen/2
        bally = x*scale * np.sin(a) + screen_height/2+polewidth/2+ballrad
        self.balltrans.set_translation(ballx, bally)
        self.poletrans.set_rotation(a)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class Simplest1DMA(BallOnPlate):
    """Simplified multi-agent 2D Ball on Plate class.

        This class implements the simplified multi-agent 2D model. It is the
        same as the simplified 2D model, but uses multiple agents.

        Note:
            This model is still experimental!! There are no guidelines for
            multi-agent systems, implemented as OpenAI gym environments yet.
            Hence, a multi-agent implementation is completely up to the user.
    """
    def __init__(self, integration_method, initial_states=[None, None], **kwargs):
        logger.warn("The implementation of the multi-agent version of the 2D ball on plate model is not complete yet!")
        super(Simplest1DMA, self).__init__(integration_method, **kwargs)
        self.M = 2 # number of agents
        self._x_max = self._plate_length 
        self._x_min = 0.0
        self._v_max = np.finfo(np.float32).max
        self._v_min = -np.finfo(np.float32).max

        self._low = np.array([self._x_min, self._v_min])
        self._high = np.array([self._x_max, self._v_max])

        self.action_space = []
        self.observation_space = []
        for _ in range(self.M):
            self.action_space.append(spaces.Discrete(2))
            self.observation_space.append(spaces.Box(self._low, self._high, dtype=np.float32))
        
        self._state = [None, None]
        self.reset(initial_states)

    def _EoM(self, state, action):
        """Equations of motion for simplified multi-agent 2D Ball on Plate model.

            See generic method of parent class for more information.

            Args:
                state (np.array): Numpy array of the current state.
                action (np.array): Numpy array of the current action.
        """
        pos1, pos2 = action
        u = 0.5*(pos2 - pos1)
        x = state[0]
        xd = state[1]

        #EoM
        xdd = -self._gravity*self._ball_mass/self._M * u/self._plate_length
        return np.array([xd, xdd])

    def reset(self, initial_states=[None, None]):
        """Reset method for simplified multi-agent 2D Ball on Plate model.
        
            See generic method of parent class for more information.

            Args:
                initial_state (np.array): Numpy array of the state, which
                    should be set as the inital state. If None, the position
                    is chosen randomly, while the velocity is always zero.
                    This is done for each agent individually.

            Returns:
                state (np.array): Numpy array containing the inital state.
        """
        super(Simplest1DMA, self)._reset()
        self._input = [0]
        for m in range(self.M):
            if initial_states[m] is None:
                state = self.observation_space[m].sample()
                self._state[m] = np.array([state[0], 0.0])
            else:
                self._state[m] = initial_states[m]
        return self._state

    def _reward(self, done, state):
        """Reward method for simplified multi-agent 2D Ball on Plate model.
        
            This method returns the reward given the state and if the episode
            is finished or not. A reward of 1000 is given if the episode was
            successful, i.e. the episode ended with the ball standing still in
            the middle of the plate. If the ball falls off the plate, the reward
            is -1000 and if the episode finishes somewhere on the plate but not
            the middle the reward is -100. Additionally, in each time step an
            reward of 1 is given if the ball is moving towards the middle of the
            plate and -1 if it moves in the opposite direction.

            Args:
                done (bool): True if the episode is finished, False otherwise.
                state (np.array): Numpy array containing the current state.

            Returns:
                reward (float): Reward of the current time step.
        """
        x,v = state
        if not done:
            reward = -np.sign(x-0.5)*np.sign(v)
        else:
            if 0.498 <= x <= 0.502 and np.abs(v) < 0.01:
                reward = 1000
            elif self._t > 10:
                reward = -100
            else:
                reward = -1000
        return reward

    def step(self, actions):
        """Step method of the simplified multi-agent 2D Ball on Plate model.
        
            This method implements the standard gym environment step method.
            The method computes the next state and reward, given the action.
            The episode is terminated if the ball fall of the plate, the ball
            stops in the middle of the plate or if the time exceeds 5 seconds.
            For more information about the reward see the reward method.

            See generic method of parent class for more information.

            Args:
                action (np.array): Numpy array of the action to apply to the
                    environment.

            Returns:
                state (np.array): Numpy array of the next state.
                reward (float): Reward for the step taken.
                done (bool): True, if the episode is terminated. False, otherwise.
                info (dict): Optional info dictionary.
        """
        if not self._done:
            dt = self._dt
            t = self._t
            self._state = self._integrate(self._EoM, state, action, dt)
            self._t += self._dt

            x, v = self._state

            done = not self.observation_space.contains(self._state) \
                    or self._t > 10 \
                    or ( 0.498 <= x <= 0.502 and np.abs(v) < 0.01)
            self._done = bool(done)
        else:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        reward = self._reward(self._done, self._state)
        return self._state, reward, self._done, {}

class Full1D(BallOnPlate):
    """Full 2D Ball on Plate class.

        This class implements the full 2D model. The 2D model corresponds
        to a Ball on a Bar model. For this model, no dynamics are simplified.
        See the report for further details on the model.

        Note:
            This class only implements the derived equations of motions, but nothing
            more. Hence, for now this model can only use the simulate method.
    """
    def __init__(self, arg):
        logger.warn("Only equations of motion are implemented for now! You can only use simulate method.")
        super(Full1D, self).__init__()

    def _EoM(self, state, action):
        """Equations of motion for full 2D Ball on Plate model.

            See generic method of parent class for more information.

            Args:
                state (np.array): Numpy array of the current state.
                action (np.array): Numpy array of the current action.
        """
        x = state[0]
        xd = state[1]
        a = state[2]
        ad = state[3]
        T = action

        #EoM
        xdd = 1/self._M * (self._ball_mass * x * ad**2 - self._gravity*self._ball_mass * np.sin(a))
        add = 1/(self._I+self._ball_mass*x**2)*(T
                                              - 2*self._ball_mass*x*xd*ad
                                              - self._gravity*np.cos(a)*(self._ball_mass*x
                                                                        + self._plate_mass*self._plate_length/2)
                                             )
        return np.array([xd, xdd, ad, add])

class Full2D(BallOnPlate):
    """Full 3D Ball on Plate class.

        This class implements the full 3D model as derived for this Master's
        thesis. See the report for further details on the model.

        Note:
            This class only implements the derived equations of motions, but nothing
            more. Hence, for now this model can only use the simulate method.
    """
    def __init__(self, arg):
        logger.warn("Only equations of motion are implemented for now! You can only use simulate method.")
        super(Full2D, self).__init__()

    def _EoM(self, state, action):
        """Equations of motion for simplified 3D Ball on Plate model.

            See generic method of parent class for more information.

            Args:
                state (np.array): Numpy array of the current state.
                action (np.array): Numpy array of the current action.
            
            Note:
                WARNING THERE IS STILL A PROBLEM WITH add AND bdd!
        """
        x = state[0]
        xd = state[1]
        y = state[2]
        yd = state[3]
        a = state[4]
        ad = state[5]
        b = state[6]
        bd = state[7]
        Ta = u[1]
        Tb = u[2]

        #EoM
        xdd = 1/self._M * (self._ball_mass * x * ad**2 + self._ball_mass * bd * ad * y
                          - self._gravity*self._ball_mass * np.sin(a))
        ydd = 1/self._M * (self._ball_mass * y * bd**2 + self._ball_mass * bd * ad * x
                          - self._gravity*self._ball_mass * np.sin(b))
        add = 1/(self._I+self._ball_mass*x**2)*(Ta
                                              - self._ball_mass*(2*x*xd*ad + bdd*x*y + bd*xd*y + bd*x*yd)
                                              - self._gravity*np.cos(a)*(self._ball_mass*x
                                                                        + self._plate_mass*self._plate_length/2)
                                             )
        bdd = 1/(self._I+self._ball_mass*y**2)*(Tb
                                              - self._ball_mass*(2*y*yd*bd + add*x*y + ad*xd*y + ad*x*yd)
                                              - self._gravity*np.cos(b)*(self._ball_mass*y
                                                                        + self._plate_mass*self._plate_width/2)
                                             )
        return np.array([xd, xdd, yd, ydd, ad, add, bd, bdd])

class MassSpringDamper(gym.Env):
    """Mass-Spring-Damper class.

        This class implements the classical Mass-Spring-Damper model
        as a multi-agent system. The model is based on the OpenAI
        gym environment class and implements additional methods, e.g.
        simulate, to simulate the system forward in time, and the
        get_system_matrices method, to get the relevant matrices for
        the invariance-based safety framework.

        Note:
            In contrast to the Ball on Plate models, the dynamics are
            stored in state-space form, i.e. matrices, instead of multiple
            ordinary differential equations.
    """
    def __init__(self,
                 integration_method="default",
                 N=1,
                 dt=0.01, #s
                 m=1,     #kg
                 k=10,    #kg/s^2
                 d=5     #kg/s
                ):
        """Initialize Mass-Spring-Damper class.
        
            This method initalizes all physical quantities and builds the
            integration method. The masses are always connected by the springs
            and dampers in a line. If another arrangement is needed, the class
            needs to be modified.

            Args:
                integration_method (str): Name of integrator method, as registred
                    in the INTEGRATORS dictionary.
                N (int): Number of agents, i.e. masses, to use.
                dt (float): Time discretization in seconds.
                m (float or list): Mass(es) of the agents in kilograms. If float,
                    all the masses get the same value m; if list, the masses
                    are defined as in the given list. List needs N entries!
                k (float or list): Spring constant(s) of the springs connecting
                    the agents in  [N/m]. If float, all spring constants get the
                    same value k; if list, the spring constants are defined as in
                    the given list. List needs N-1 entries!
                d (float or list): Damping coefficient(s) of the dampers connecting
                    the agents in  [N*s/m]. If float, all damping coefficients get
                    the same value d; if list, the damping coefficients are defined
                    as in the given list. List needs N-1 entries!
        """
        self._x_min = -1  #m
        self._x_max = 1   #m
        self._v_min = -3  #m/s
        self._v_max = 3   #m/s
        self._u_min = -2  #N
        self._u_max = 2   #N
        
        self.N = N
        
        if type(m) is int:
            self._m = [m] * N
        else:
            assert len(m) == N
            self._m = m
        
        if type(k) is int:
            self._k = [k] * (N-1)
        else:
            assert len(k) == N-1
            self._k = k
        
        if type(d) is int:
            self._d = [d] * (N-1)
        else:
            assert len(d) == N-1
            self._d = d
        
        Arows = []
        Brows = []
        if N == 1:
            self._A = np.array([[0, 1], [-k/m, -d/m]])
            self._B = np.array([[0],[1/m]])
        else:
            for i in range(N):
                if i == 0:
                    m = self._m[i]
                    d = self._d[i]
                    k = self._k[i]
                    diag = np.array([[0, 1], [-k/m, -d/m]])
                    odiag = np.array([[0,0],[k/m, d/m]])
                    Arows.append(np.hstack([diag, odiag, np.zeros((2,2*N-2*2))]))
                elif i == N-1:
                    m = self._m[i]
                    d = self._d[i-1]
                    k = self._k[i-1]
                    diag = np.array([[0, 1], [-k/m, -d/m]])
                    odiag = np.array([[0,0],[k/m, d/m]])
                    Arows.append(np.hstack([np.zeros((2,2*N-2*2)), odiag, diag]))
                else:
                    m = self._m[i]
                    d1 = self._d[i]
                    k1 = self._k[i]
                    d0 = self._d[i-1]
                    k0 = self._k[i-1]
                    odiag0 = np.array([[0,0],[k0/m, d0/m]])
                    diag = np.array([[0, 1], [-(k0+k1)/m, -(d0+d1)/m]])
                    odiag1 = np.array([[0,0],[k1/m, d1/m]])
                    Arows.append(np.hstack([np.zeros((2,(i-1)*2)), odiag0, diag, odiag1, np.zeros((2,(N-i-2)*2))]))
                Brows.append(np.kron(np.eye(1,N,k=i), np.array([[0],[1/m]])))
            self._A = np.vstack(Arows)
            self._B = np.vstack(Brows)
    
        self._dt = dt

        self._input = None
        self._done = False
        self.viewer = None
        self._integrate = Integrator(integration_method)

        self.seed()
        
        self._low = np.array([self._x_min, self._v_min])
        self._high = np.array([self._x_max, self._v_max])
        
        self.observation_space = spaces.Box(np.tile(self._low, N), np.tile(self._high, N), dtype=np.float32)
        self.action_space = spaces.Box(np.tile(self._u_min, N), np.tile(self._u_max, N), dtype=np.float32)
        
        self.reset()
    
    def _EoM(self, state, action):
        """Equations of motion.
        
            This method implements the continuous time equations of motion of
            the Mass-Spring-Damper model. Here, the equations of motion are given
            as a simple matric multiplication.

            Args:
                state (np.array): Numpy array of the current state.
                action (np.array): Numpy array of the current action.

            Retruns:
                next_state (np.array): Numpy array of the next state under the
                    given action.
        """
        return self._A@state + self._B@action

    def _model(self, t, y):
        """Model wrapper for simulation.
        
            This method implements a wrapper for the simulate method. The solve_ivp
            method of scipy doesn't allow the use of an input. This wrapper adds this
            functionality, by querying the _input method in each time step to get
            the current action value.

            Args:
                t (np.array): Numpy array of the current time.
                y (np.array): Numpy array of the current state.

            Note:
                The naming is adapted to the scipy.integrate module.
        """
        action = self._input(t)
        return self._EoM(y, action)
    
    def _reward(self, done, state):
        """Reward function.

            This method implements the reward function of the Mass-Spring-Damper
            model. However, the model wasn't used in combination with a
            reinforcement learning algorithm yet. Hence, the reward is still zero
            and no method is implemented yet. The reason for this lies in the
            bad scaling of Q-Learning for multi-agent systems.
        """
        raise NotImplementedError

    def close(self):
        """Close environment renderer.
        
            This method closes the environment renderer. This is a standard
            method of all OpenAI gym environments, see gym documentation for further
            information.

            Args:
                None.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        """Environment renderer.
        
            This method implements a renderer of the environment, in order to allow for
            a visual output to the screen. This is a standard method of all OpenAI gym
            environments, see gym documentation for further information.

            This method needs to be implemented yet.

            Args:
                mode (str): Render mode.

            Raises:
                NotImplementedError: If this method is called.
        """
        raise NotImplementedError
    
    def reset(self, initial_state=None):
        """Environment reset.
        
            This method resets the environment. This method should be called after
            an episode is finished. This is a standard method of all OpenAI gym
            environments, see gym documentation for further information.

            Here, the position is sampled randomly from the observation space,
            while the velocity is always set to zero. Unless the argument
            initial_state is not None.
            
            Args:
                initial_state (np.array): Numpy array of the state, which
                    should be set as the inital state. If None, the position
                    is chosen randomly, while the velocity is always zero.

            Returns:
                state (np.array): Numpy array containing the inital state.
        """
        self._t = 0
        self._input = None
        self._done = False
        self._input = None
        if initial_state is None:
            state = self.observation_space.sample()
            state[1::2] = np.zeros(self.N)
            self._state = state
        else:
            self._state = initial_state
        return self._state

    def simulate(self, t_span, init_state, u, solver='RK45', dense_output=True, **kwargs):
        """Method to simulate the model forward in time.
        
            This method allows to simulate the model forward in time, given
            an input. The state of the model is not affected, hence calling
            simulate in between calls to step, has no effect!

            The interface is adapted from the solve_ivp method of the
            scipy.integrate module. For more information have a look at the
            scipy documentation.

            Args:
                t_span (2-tuple of floats): Integration interval (t0, tf).
                init_state (np.array): Numpy array of the initial state.
                u (fun): Callable function, returning the input of the current
                    time step. Should accept time as an argument.
                solver (str): Wether to use the internal integrator ("internal")
                    or to use the specified solver from scipy.integrate.
                dense_output (bool): Wether to return the complete time vector
                    or use a sparse representation.
        """
        if solver == 'internal':
            Iteration = namedtuple('Iteration', ('t', 'y', 'u'))
            output = []
            state = init_state
            t = t_span[0]
            output.append(Iteration(t=t, y=state, u=None))
            for i in range(int(t_span[1]//self._dt)):
                action = u(t)
                state = self._integrate(self._EoM, state, action, self._dt)
                t += self._dt
                output.append(Iteration(t=t, y=state, u=action))
            output = Iteration(*zip(*output))
            return output
        else:
            self._input = u
            return odesolver(self._model, t_span, init_state, method=solver, dense_output=dense_output, **kwargs)
    
    def seed(self, seed=None):
        """Environment seed.
        
            This method sets the random seed of the environment. This is a standard
            method of all OpenAI gym environments, see gym documentation for further
            information.

            Args:
                seed (int): Random seed.

            Returns:
                seed (int): Random seed.

            Note:
                This sets the random generator instance as an attribute of the class.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        """Environment step.
        
            This method implements the standard gym environment step method.
            The method computes one step of the model, given the current
            state stored in the class and the provided action. See gym documentation
            for further information.

            Here, the episode is terminated if the state leaves the pre-defined
            state constraints, defined by observation_space. Or if the time exceeds
            10 seconds.

            Args:
                action (np.array): Numpy array of the action to apply to the
                    environment.

            Returns:
                state (np.array): Numpy array of the next state.
                reward (float): Reward for the step taken. For now, this is always
                    zero.
                done (bool): True, if the episode is terminated. False, otherwise.
                info (dict): Optional info dictionary.
        """
        if not self._done:
            state = self._state
            dt = self._dt
            t = self._t
            self._input = action
            self._state = self._integrate(self._EoM, state, action, dt)
            self._t += self._dt

            done = not self.observation_space.contains(self._state) \
                   or self._t > 10
            self._done = bool(done)
        else:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        reward = 0
        return self._state, reward, self._done, {}
    
    def get_system_matrices(self, uncertainty, k=True, d=True):
        """Return system matrices.
        
            This method returns the matrices defining the system, required
            by the invariance-base safety framework.

            Args:
                uncertainty (float): Uncertainty in percent to apply to the
                    uncertain parameters.
                k (bool): Wether the spring constant is uncertain or not.
                d (bool): Wether the damping coefficient is uncertain or not.

            Returns:
                A (list): List of state matrices, given the uncertain parameters.
                    The list only contains the extremal parameter realizations.
                B (list): List of input matrices, given the uncertain parameters.
                    The list only contains the extremal parameter realizations.
                Ax (np.array): State constraint matrix.
                bx (np.array): State constraint vector.
                Au (np.array): Input constraint matrix.
                bu (np.array): Input constraint vector.
                dims (list): List of the dimensions of each agent in the system.
                G (np.array): Adjacency matrix of the communication graph as
                    numpy array.
        """
        N = self.N
        
        mask = np.eye(N) + np.eye(N,k=1) + np.eye(N,k=-1)
        deltak = self._A*np.kron(mask,np.array([[0,0],[1,0]]))*uncertainty
        deltad = self._A*np.kron(mask,np.array([[0,0],[0,1]]))*uncertainty
        if k:
            ks = [deltak, -deltak]
        else:
            ks = [deltak*0.0]
        if d:
            ds = [deltad, -deltad]
        else:
            ds = [deltad*0.0]
        
        A = []
        for k in ks:
            for d in ds:
                A.append(np.eye(2*N)+self._dt*(self._A + k + d))

        B = [self._dt*self._B]
        
        Axdiag = np.array([[1,0], [-1, 0], [0,1], [0,-1]])
        bxdiag = np.ones((4,1)) * np.abs(np.array([self._x_max, self._x_min, self._v_max, self._v_min]).reshape(4,1))
        Audiag = np.array([[1],[-1]])
        budiag = np.ones((2,1))*np.abs(np.array([self._u_max, self._u_min]).reshape(2,1))
    
        Ax = np.kron(np.eye(N), Axdiag)
        bx = np.kron(np.ones((N,1)), bxdiag)
        Au = np.kron(np.eye(N), Audiag)
        bu = np.kron(np.ones((N,1)), budiag)
        G = np.eye(N) + np.eye(N, k=-1) + np.eye(N, k=1)
        dims = np.ones(N, dtype=np.int) * 2
        
        return A, B, Ax, bx, Au, bu, dims, G

    """INFO"""
    def info(self):
        """Environment info.
        
            This method prints the info and settings of the model to the logging stream.

            Args:
                None.
        """
        logger.info("MODEL INFO")
        logger.info(self.__class__)
        if self.__doc__ is None:
            logger.info("No documentation found.")
        else:
            logger.info(self.__doc__)
        logger.info("INTEGRATOR INFO")
        logger.info(self._integrate._method.__name__)
        if self._integrate._method.__doc__ is None:
            logger.info("No documentation found.")
        else:
            logger.info(self._integrate._method.__doc__)
        return

    def get_state(self):
        """Print current state.
        
            This method prints the current time and state to the logging stream.

            Args:
                None.
        """
        logger.info("Time {0}".format(self._t))
        logger.info("State {0}".format(self._state))
        return {"time": self._t, "state": self._state}

""" Registered model classes. """
MODELS = {'default': Simplest1D,
          'BoPSimple1D-v0': Simplest1D,
          'BoPUncertain1D-v0': Uncertain1D,
          'BoPFull1D-v0': Full1D,
          'BoPFull2D-v0': Full2D,
          'MABoP1D-v0': Simplest1DMA,
          'MassSpringDamper-v0': MassSpringDamper}
