import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.preprocessing import StandardScaler
from rl_framework.Utils import compute_covariance_matrix_mean, reset_specific_keys, setLogger, create_feature_matrix, change_num_initial_particles, create_seeded_tracking_file
import random
from rl_framework.Elegant import ElegantWrapper
import math
import os

class ACCElegantEnvironment(gym.Env):
    def __init__(self,stage = None,n_bins= 5, init_num_particles= None, results_path= ' ', input_beamline_file= "machine.lte", input_beam_file= "track", beamline_name= 'machine', output_beamline_file='updated_machine.lte', elegant_input_filename='elegant_input.lte', reset_specific_keys_bool= True, logger=None, file_handler=None, elegant_path= "/Users/anwar/Downloads/sdds/darwin-x86/", sddsPath= "/Users/anwar/Downloads/sdds/defns.rpn", override_dynamic_command= False, overridden_command= " ", seed=0):
        self.override_dynamic_command= override_dynamic_command
        self.overridden_command= overridden_command
        self.seed = int(seed)
        self.input_file_path = input_beamline_file
        self.base_input_beam_file = input_beam_file
        self.output_file = output_beamline_file
        self.elegant_input_filename = elegant_input_filename
        self.beamline_name = beamline_name
        self.elegantPath = elegant_path
        self.sddsPath = sddsPath
        self.results_path= results_path
        self.inputs_dir = f"inputs_{self.seed}"
        self.n_bins= n_bins
        self.initial_reward= 1
        self.init_num_particles= init_num_particles #we are solving for one number of particles.
        self.input_beam_file = create_seeded_tracking_file(
            self.base_input_beam_file,
            self.elegant_input_filename,
            self.results_path,
            self.seed
        )
        self.wrapper = ElegantWrapper(self.input_file_path, self.input_beam_file, self.beamline_name, self.output_file, elegant_path= self.elegantPath, sddsPath= self.sddsPath, results_path= self.results_path, elegant_input_filename= self.elegant_input_filename, overrid_dynmaic_commnad=self.override_dynamic_command, overrideen_command= self.overridden_command, seed=self.seed)
        
        '''
        print("#############$$$$$$$$$$$$$$$$$$$$$ Env DEBUG $$$$$$$$$$$$$$##############")
        
        print("###### from Environment ######")
        print("Results Path: ", self.results_path)
        print("override_dynamic_command:  ", override_dynamic_command)
        print("overridden_command:  ", overridden_command)
        print("self.wrapper.overrid_dynmaic_commnad:  ", self.wrapper.overrid_dynmaic_commnad)
        print("self.wrapper.overrideen_command:  ", self.wrapper.overrideen_command)
        
        print("###### Outside Environment ######")
        print("#############$$$$$$$$$$$$$$$$$$$$$ Env DEBUG $$$$$$$$$$$$$$##############")
        '''
        
        self.reset_specific_keys_bool = reset_specific_keys_bool
        self.wrapper.reset_specific_keys_bool = self.reset_specific_keys_bool
        self.max_num_of_vars=  len(self.wrapper.chroneological_order_controllable_vars)
        self.max_num_of_states = self.wrapper.max_itteration
        if stage != None and stage < self.max_num_of_vars:
            self.stage = stage  # Stage for curriculum learning
        else:
            self.stage= None
        self.stage_mask = None  # Mask for controlling variables based on the stage
        self.observation = None
        self.variables = self.wrapper.chroneological_variables
        self._set_action_space(self.variables)
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_shape,), dtype=np.float32)
        self.logger = logger
        self.file_handler = file_handler
        

        # Initialize the stage mask if stage is not None
        if self.stage is not None:
            self.stage_mask = self._get_stage_mask()
        #rest the self.iteration to zero because _get_stage_mask changes it, we can also call reset() outside to get the same effect.
        self.iteration= 0

    def _set_action_space(self, variables):
        """Set the action space based on the list of variables. set correct max, min ranges for each variable"""
        #keep this action space to be used for convert vars and other functions , but it will not reflect the actual action space.
        action_low = []
        action_high = []
        
        for var in variables:
            if "K1" in var:
                action_low.append(-20)
                action_high.append(20)
            elif any(x in var for x in ["VKICK", "HKICK", "FSE"]):
                action_low.append(-0.005)
                action_high.append(0.005)
            else:
                # Default range for other variables (if any)
                action_low.append(-1)
                action_high.append(1)

        action_low = np.array(action_low, dtype=np.float64)
        action_high = np.array(action_high, dtype=np.float64)
    
        self.action_space_fake = spaces.Box(low=action_low, high=action_high)
        self.action_space_fake.n = self.action_space_fake.low.size

        #Real action space, this one to be used for the actual environment and it will also be used to define the shape of the output of the policy network.
        action_low=   np.array([-20, -0.005, -0.005, -0.005], dtype=np.float64)
        action_high= np.array([20, 0.005, 0.005, 0.005], dtype=np.float64)

        self.action_space = spaces.Box(low=action_low, high=action_high)
        self.action_space.n = self.action_space.low.size
    
    def _convert_variables(self, values):
        """Convert values from the range [-1, 1] to the correct range based on the variable names."""
        converted_variables = []
        for i, value in enumerate(values):
            range_min = self.action_space.low[i]
            range_max = self.action_space.high[i]
            converted_variable = value * (range_max - range_min) / 2 + (range_max + range_min) / 2
            converted_variables.append(converted_variable)
            
        return np.round(converted_variables,4)
    
    def _get_stage_mask(self):
        """
        Generate a stage mask based on the current stage.
        The stage mask determines which variables are controlled in the current stage.

        Returns:
        np.array: A binary mask indicating the variables to control.
        """
        stage_mask = np.zeros(len(self.variables))
        count = 0

        # Iterate through the stages and accumulate the count of variables to control
        for i in range(self.stage):
            self.iteration = i
            count += self._check_number_of_variables_to_be_set_at_this_iteration()
            stage_mask[:count] = 1

        return stage_mask
    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        if seed is not None and int(seed) != self.seed:
            self.seed = int(seed)
            self.inputs_dir = f"inputs_{self.seed}"
            self.input_beam_file = create_seeded_tracking_file(
                self.base_input_beam_file,
                self.elegant_input_filename,
                self.results_path,
                self.seed
            )
        if self.init_num_particles ==None:
            self.init_num_particles= random.randint(500, 50000)
            old, self.initial_number_of_particles = change_num_initial_particles(self.input_beam_file+".ele", self.init_num_particles)
        else:
            old, self.initial_number_of_particles = change_num_initial_particles(self.input_beam_file+".ele", self.init_num_particles)
        # initial number of partciles is actually new and it should be equal to self.initial_reward
        self._set_initial_number_of_particles(self.initial_number_of_particles)
        self.done = False
        self.reward = 0.0
        #self.wrapper= ElegantWrapper(self.input_file_path, self.input_beam_file,self.beamline_name, self.output_file)
        self.wrapper = ElegantWrapper(self.input_file_path, self.input_beam_file, self.beamline_name, self.output_file, elegant_path= self.elegantPath, sddsPath= self.sddsPath, results_path= self.results_path, elegant_input_filename= self.elegant_input_filename, overrid_dynmaic_commnad=self.override_dynamic_command, overrideen_command= self.overridden_command, seed=self.seed)
        self.wrapper.reset_specific_keys_bool = self.reset_specific_keys_bool
        self.iteration = 0
        self.previous_mask_len = 0
        self.mask_len = 0
        self.actions_ = np.ones((len(self.variables),))
        # Run the simulation with the converted action values
        values=np.zeros((len(self.variables)))
        elegant_input, success, dict_vars = self.wrapper.run_elegant_simulation(values)
        # Get the results of the simulation
        observations, reward, output_file, done = self.wrapper.get_results(self.initial_number_of_particles)

        if reward != 0: # to make sure we don't devide by zero.
            self.initial_reward= reward
            self.number_of_particle_prev= self.initial_reward

        # Preprocess the observation data
        observation = observations

        """get the correct shape of the state"""
        
        """
        Returns 1D numpy array with:
        - Median, IQR, 10th, 90th percentiles for x, y, xp, yp (4 * 4 = 16)
        - 2D histogram bin counts for (x, y) (n_bins * n_bins = 25 for n_bins=5)
        - Number of particles (1)
        - Fraction of particles inside ellipse (1)
        - Covariance matrix upper triangle for x, y, xp, yp (10 elements)
        - Ellipse parameters a_previous, b_previous and a_next, b_next (4)
        Total size: 16 + 25 + 1 + 1 + 10 + 2 = 57
        """
        self.observation_shape = 16 + self.n_bins**2 + 1 + 1 + 10 + 4  # 55 elements. NOT DYNAMIC

        self.observation = observation #np.array([observation])
        return self.observation, {}
    
    def _set_initial_number_of_particles(self,new):
        self.initial_number_of_particles = new

        
    def _check_number_of_variables_to_be_set_at_this_iteration(self):
        """Check the number of variables to be set at this iteration."""
        #current_var = self.variables[self.iteration]
        count= 0
        if self.iteration< len(self.wrapper.chroneological_order_controllable_vars):
            current_var=self.wrapper.chroneological_order_controllable_vars[self.iteration]
            #current_var = self.variables[self.previous_mask_len]
            #base_name = current_var.replace("K1", "").replace("VKICK", "").replace("HKICK", "").replace("FSE", "")
            #count = sum(1 for var in self.variables if base_name in var)
            #count = sum(1 for var in self.variables if current_var in var)
            for var in self.variables:
                base_name = var.replace("K1", "").replace("VKICK", "").replace("HKICK", "").replace("FSE", "")
                if current_var == base_name:
                    count+=1
                
        return count
    
    def _get_mask(self, mask_len):
        """Generate a mask for the current iteration.
        len(variables)= 10 
        mask_len=3
        output= [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        """
        mask = np.zeros(len(self.variables))
        mask[:mask_len] = 1
        return mask
    
    def _get_action_mask(self, count):
        mask= np.zeros(self.action_space.n)
        if count == 3:
            mask[:count] = 1
        elif count==1:
            mask[-count] = 1

        
        return mask

    def _get_new_action(self, count, action):
        new_action = None
        if count == 3:
            new_action= np.zeros(3)
            new_action= action[:count]  
        elif count==1:
            new_action= np.zeros(1)
            new_action= action[-count] 

        return new_action

    def _correct_action(self, action):
        """
        Ensure that only the new variables corresponding to the current iteration are updated,
        while keeping the previously set variables unchanged.

        Parameters:
        action (np.array): The action values to be applied.

        Returns:
        np.array: The combined action values with the new variables updated and previous variables unchanged.

        Example:
        inputs: 
        len(variables)= 10 
        mask_len=5
        previous_mask_len= 3

        outputs:
        action_mask= [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.]
        previous_actions_mask= [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]
        """
        # Create a mask for the current iteration's variables
        action_mask = np.zeros(len(self.variables))
        #action_mask[self.previous_mask_len:self.mask_len] = 1
        action_mask[self.previous_mask_len:self.mask_len]= action
        
        # Apply the mask to the new action values
        ###new_action = action * action_mask

        # Create a mask for the previously set variables
        previous_actions_mask = np.zeros(len(self.variables))
        previous_actions_mask[:self.previous_mask_len] = 1

        # Combine the previous actions with the new action values
        #previous_Actions = (previous_actions_mask * self.actions_) + new_action #!!!!
        previous_Actions = (previous_actions_mask * self.actions_) + action_mask
        return previous_Actions
    

    def _log_episode(self, info):
        """Log the episode information."""
        ep_info= {
            'reward': info["reward"],
            'done': info["done"],
            'itteration': info["itteration"],
            'current_element': info["output_file"],
            'dict_vars': info["dict_vars"],
            'initial_number_of_particles': info["initial_number_of_particles"],
            'number_of_particles': info["number_of_particles"]
        }
        self.logger.writerow(ep_info)
        self.file_handler.flush()

    def step(self, action, convert=False):
        """
        Perform a training step in the environment with stage learning applied.
        """

        info = {}
        if convert:
            action = self._convert_variables(action)
        # Determine the number of variables to set in this iteration
        num_of_vars_iteration = self._check_number_of_variables_to_be_set_at_this_iteration()
        self.mask_len = self.previous_mask_len + num_of_vars_iteration
        self.action_mask = self._get_action_mask(num_of_vars_iteration)
        new_actions= self._get_new_action(num_of_vars_iteration, action)
        self.mask = self._get_mask(self.mask_len)
        

        self.actions_ = self._correct_action(new_actions)

        # Apply the stage mask to restrict actions based on the current stage
        if self.stage_mask is not None:
            self.actions_ = self.actions_ * self.stage_mask

        self.previous_mask_len = self.mask_len
        elegant_input, success, dict_vars = self.wrapper.run_elegant_simulation(self.actions_) 
        
        '''
        print("@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(elegant_input)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@")
        '''

        if not os.path.exists(self.inputs_dir):
            os.makedirs(self.inputs_dir)
        
        try:
            # Save the elegant input for debugging
            with open(os.path.join(self.inputs_dir, "elegant_input" + str(self.iteration) + ".txt"), "w") as file:
                file.write(elegant_input)
        except TypeError as e:
                    print(f"Warning: {e}")

        # Get the results of the simulation
        observations_dataframe, reward, output_file, done = self.wrapper.get_results(self.initial_number_of_particles)

        #### reward= (1000* reward)/ self.initial_reward #updating reward
        self.number_of_particle_curr= reward
        
        reward= self.number_of_particle_curr/ self.initial_reward
        

        #reward= (reward)/ self.initial_reward
        #print("reward <= 3 ==", reward <= 3)
        self.done= done
        if self.number_of_particle_curr <= 3:   
            self.done = True  # We lost all particles
            reward -= math.sqrt(abs((self.max_num_of_vars**2)- ((self.iteration+1)**2)))*(1/self.max_num_of_vars) #add penality for losing all particles before the end of the beamline
            #reward = -1 * math.sqrt(abs((self.max_num_of_vars**2)- ((self.iteration+1)**2)))*(100/self.max_num_of_vars) #add penality for losing all particles before the end of the beamline


        #also reward >3:  fix this
        elif self.number_of_particle_curr > 3:
            reward *=  self.number_of_particle_curr/ self.number_of_particle_prev

            #this means reward= (curr_particles/ inital_particls)+ (curr_particles/prev_particles)

            self.number_of_particle_prev= self.number_of_particle_curr
            if output_file == "final_WP":
                #reward = round((self.iteration+1) * reward, 3)
                ####reward = round(1 * reward, 3) # add extra one percent
                #if output_file ==  "final_WP":#reached the end without losing all particles
                    #reward += temp_particles  ## add an extra positive feedback to the reward
                self.done = True  # We reached the end of the beamline

        self.reward = reward # sum of rewards
        
        # self.reward += reward
        # ReMOVE THIS LINE
        if observations_dataframe is not None:
            #observations_dataframe.to_csv('observs/REMOVE_observations_' + str(self.iteration) + "__" + output_file + '.csv', index=False)

            # Preprocess the observation data
            
            self.observation =  observations_dataframe

            #self.reward is actual reward "independant of the number of initial particles"
            #whereas reward is the previous reward defined by previous function
        elif self.observation is None:
            # Log the error information
            with open("error_log.txt", "a") as error_file:
                error_file.write(f"Iteration: {self.iteration}, Output File: {output_file}\n")
            
        #UNDERSTANDING ITERATIONS A BIT
        #extra precautions so we don't get over the range of variables that we can set. in reality 
        #there should be two iterations one for watch points "this defines how many states we have
        #and how many watch points" the other one is for elements "this one defines how many magnets 
        #do we have" in theory they should be the same but in reality we donc't care about the first 
        #observation so stage=2 checks until WBM1_1, and sets the first three elements so these two iterations are one number apart 
        if (self.iteration< self.max_num_of_vars): 
            self.iteration += 1
        else:
            print("itteration is over the max length of the beamline")
            self.done= True 
            print("we set done to True")
        
        if self.stage == self.iteration:
            self.done= True

        #'reward', 'initial number of particles','number of particles', 'done', 'itteration','current_element','dict_vars'
        # Fill info dictionary
        info = {
            "dict_vars": dict_vars,
            "actions": self.actions_,
            "itteration": self.iteration,
            "reward": self.reward,
            "output_file": output_file,
            "done": self.done,
            "masked_action": self.actions_,
            "observations_dataframe": observations_dataframe,
            "initial_number_of_particles": self.initial_number_of_particles,
            "number_of_particles": self.wrapper.get_num_particles()
        }

        # Log the episode information
        if self.done:
            self._log_episode(info)


        return self.observation, float(self.reward), self.done, False, info
    
    def get_number_of_particles(self):
        return self.wrapper.num_particles
    
    def stage_learning(self):
        #this function is for curriculim learning, we will train to solve the first stage then make
        #our task more complicated incrementally by increasing the stages.
        pass