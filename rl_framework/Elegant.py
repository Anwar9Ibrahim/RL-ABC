# check if observs is not created then create it
import re
import numpy as np

import os
import glob
from io import StringIO
import pandas as pd
import subprocess
from rl_framework.Utils import parse_lattice_file, add_watch_points,add_final_watch_point, change_initial_content, create_dict_from_lists, process_lte_file_to_graph, remove_watch_points, reset_specific_keys, create_feature_matrix
from rl_framework.Utils import find_maxamp_for_watch_points, create_nn_representation, points_in_region, process_particle_data

import time
import holoviews as hv
from bokeh.models import HoverTool
from IPython.display import Image
from holoviews import opts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Latex, Image
import numpy as np
import plotly.io as pio
from scipy import interpolate
import platform
import math



hv.extension("bokeh")



class ElegantWrapper:
    def __init__(self, input_beamline_file, input_beam_file, beamline_name, output_beamline_file='updated_machine.lte', elegant_path= "/Users/anwar/Downloads/sdds/darwin-x86/", sddsPath= "/Users/anwar/Downloads/sdds/defns.rpn", results_path= "results/", elegant_input_filename='elegant_input.lte', overrid_dynmaic_commnad= False, overrideen_command= " ", seed=0):
        self.overrid_dynmaic_commnad= overrid_dynmaic_commnad #set to true only when you want to override the dynamic command
        self.seed = int(seed)
        self.elegant_input_filename = elegant_input_filename
        self.overrideen_command= overrideen_command #this is the default value for the override of the command. you can define the input file so overrideen command doesn't need to include the input file.
        self.input_file_path = input_beamline_file
        self.input_beam_file = input_beam_file
        self.output_file = output_beamline_file
        self.beamline_name = beamline_name
        self.results_path = results_path #
        self.observs_dir = f"observs_{self.seed}"
        self.wtach_points = []
        self.wtach_points, self.formatted_lattice, self.variables = self._preprocess_lattice_file()
        self.dict_variables = {}
        self.itteration = 0
        self.errors = {}
        self.num_particles=0
        self.dim_s = hv.Dimension('s', unit='m', label="s")
        self.dim_x = hv.Dimension('x', unit='mm', label="x", range=(-65, +65))
        self.dim_y = hv.Dimension('y', unit='mm', label="y", range=(-65, +65))
        self.chronolgical_order_watch_points = []
        self.chroneological_order_controllable_vars= []
        self.graph = self._get_chroneological_order_elements()
        self.chroneological_variables= self._order_vars()
        self.max_itteration = len(self.chronolgical_order_watch_points)
        self.Done = False  # end of the line "the last watch point" or losing all particles
        self.elegantPath= elegant_path #"/Users/anwar/Downloads/sdds/darwin-x86/" #default vaulue 
        self.sddsPath= sddsPath #"/Users/anwar/Downloads/sdds/defns.rpn" #sddsPath #
        self.reset_specific_keys_bool = False
        # Detect the operating system
        self.os_type = platform.system().lower()
        self.mag= None
        self.dfTrackCen = None
        self.dfTrackSig = None
        self.Ax_pos = None
    
        
    def _order_vars(self):
        """
        Orders variables chronologically based on controllable variables' order.
        
        Returns:
            list: A list of variables ordered according to the chronological order 
                of their base names in self.chroneological_order_controllable_vars.
        """
        chroneological_variables = []
        #variables= ['Q1L0K1', 'Q1L0HKICK', 'Q1L0VKICK', 'Q1L1K1', 'Q1L1HKICK', 'Q1L1VKICK', 'Q1L2_1K1', 'Q1L2_1HKICK', 'Q1L2_1VKICK', 'Q1L2_2K1', 'Q1L2_2HKICK', 'Q1L2_2VKICK', 'Q1L3_1K1', 'Q1L3_1HKICK','Q1L3_1VKICK', 'Q1L3_2K1', 'Q1L3_2HKICK', 'Q1L3_2VKICK', 'Q1L4_1K1', 'Q1L4_1HKICK', 'Q1L4_1VKICK', 'Q1L4_2K1', 'Q1L4_2HKICK', 'Q1L4_2VKICK', 'Q1L7K1','Q1L7HKICK', 'Q1L7VKICK', 'Q1L9K1', 'Q1L9HKICK', 'Q1L9VKICK', 'Q1L10K1', 'Q1L10HKICK', 'Q1L10VKICK', 'BM1FSE', 'BM2FSE', 'BM3FSE', 'BM4FSE']
        #chroneological_order_controllable_vars =['Q1L0', 'Q1L1', 'BM1', 'Q1L2_1', 'BM2', 'Q1L3_1', 'Q1L4_1', 'Q1L3_2', 'BM3', 'Q1L2_2', 'BM4', 'Q1L7', 'Q1L4_2', 'Q1L9', 'Q1L10']
        remaining_vars = self.variables.copy()  # Make a copy to track remaining variables
        
        # First, process variables that match the base names in chronological order
        for name in self.chroneological_order_controllable_vars:
            matching_vars = []
            # Find all variables whose base name (after cleaning) matches the current name
            for current_var in remaining_vars:
                # Remove known prefixes/suffixes to get the base name
                base_name = (current_var.replace("K1", "")
                                    .replace("VKICK", "")
                                    .replace("HKICK", "")
                                    .replace("FSE", ""))
                if base_name == name:
                    matching_vars.append(current_var)
            # Add them to the chronological list
            chroneological_variables.extend(matching_vars)
            # Remove them from the remaining variables
            for var in matching_vars:
                remaining_vars.remove(var)
        
        # Add any remaining variables that didn't match any base name (if needed)
        chroneological_variables.extend(remaining_vars)
        
        return chroneological_variables


    def sdds2df(self, sdds_file, columns="all"): 
        if columns == "all":
            command = str(self.elegantPath) + "sddsquery " + str(sdds_file) + " -columnlist"
            columns_process = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            # Check if the command was successful
            if columns_process.returncode != 0:
                raise RuntimeError(f"Error running sddsquery: {columns_process.stderr}")
            
            # Access the stdout attribute to get the output as a string
            columns = columns_process.stdout.splitlines()
        elif isinstance(columns, list):
            # If columns is provided as a list, use it directly
            pass
        else:
            raise ValueError("Invalid value for 'columns'. Must be 'all' or a list of column names.")
        
        # Join the columns into a single string with commas
        col_str = "-col=" + ",".join(columns)
        command = str(self.elegantPath + "sdds2stream " + sdds_file + " " + col_str + " -pipe=out")
        
        # Run the sdds2stream command
        out = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Check if the command was successful
        if out.returncode != 0:
            raise RuntimeError(f"Error running sdds2stream: {out.stderr}")
        
        # Use the stdout attribute to get the command output
        DATA = StringIO("\n".join(out.stdout.splitlines()))
        df = pd.read_csv(DATA, names=columns, sep='\s+')
        
        return df
    
    
    def _preprocess_lattice_file(self, temp_output_file="temp_modified_machine.lte"):
        """Preprocesses the lattice file by adding watch points and modifying initial content."""
        #remove all watch points that we didn't add from the original file
        remove_watch_points(self.input_file_path)
        # Add strategic watch points to the lattice file
        add_watch_points(self.input_file_path, "modifide_machine_no_finalWP.lte", results_path=self.results_path)
        #add final watch point to the end of the beamline
        add_final_watch_point("modifide_machine_no_finalWP.lte",self.output_file, self.beamline_name, results_path=self.results_path)
        # Get a list with the watch points
        u_parsed_data = parse_lattice_file(self.output_file)
        wtach_points = [key for key, value in u_parsed_data.items() if value['type'] == "watch point"]
        # Modify the initial content of the lattice file
        formatted_lattice, variables = change_initial_content(self.output_file, temp_output_file)
        #print(formatted_lattice)
        return wtach_points, formatted_lattice, variables

    def _replace_variables(self, variables, input_string):
        """
        Replaces placeholders like {variables['key']} in the input_string
        with corresponding values from the variables dictionary.

        :param variables: Dictionary containing variable names as keys and their values
        :param input_string: String with placeholders to be replaced
        :return: Modified string with placeholders replaced by corresponding values
        """
        # Pattern to match placeholders in the format {variables['key']}
        pattern = re.compile(r"\{variables\['(.*?)'\]\}")

        # Function to replace each match with the corresponding value from the dictionary
        def replacer(match):
            key = match.group(1)  # Extract the key inside the placeholder
            return str(variables.get(key, match.group(0)))  # Replace with value or keep original if key not found

        # Substitute all placeholders in the input_string
        return pattern.sub(replacer, input_string)

    def _get_elegant_input(self, values):
        self.dict_variables = create_dict_from_lists(self.chroneological_variables, values)
        # Check if the self.formatted_lattice is not empty
        if self.reset_specific_keys_bool:
            # If the self.reset_specific_keys is True, reset the specific keys in the dictionary
            self.dict_variables = reset_specific_keys(self.dict_variables)
            
        if self.formatted_lattice:
            # If the self.formatted_lattice is not empty, replace the variables with the values
            elegant_input = self._replace_variables(self.dict_variables, self.formatted_lattice)
            return elegant_input
        else:
            # If empty, return an error message
            print("Error: The lattice file is empty.")
            return None

    '''def _setup_results_folder(self):
        """Create and clear the results folder."""
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)
        for f in glob.glob(self.results_path + '*'):
            os.remove(f)'''
    def _setup_results_folder(self):
        """Create and clear the results folder, keeping .png files."""
        '''
        print("#########  Debug results file from elegant ##########")
        print(self.results_path)
        print("#########  Debug results file from elegant ##########")
        '''
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)
        for f in glob.glob(os.path.join(self.results_path, '*')):
            if not f.lower().endswith('.png'):
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"Warning: Could not remove {f} - {e}")

    def run_elegant_simulation(self, values):
        # Create results folder if it does not exist
        self._setup_results_folder()
        # Create the elegant input file
        elegant_input = self._get_elegant_input(values)
        '''
        print("#############$$$$$$$$$$$$$$$$$$$$$ DEBUG $$$$$$$$$$$$$$##############")
        if elegant_input is None:
            print("None")
        else:
            print("Not None")
        print("Elegant input file")
        print(elegant_input)
        
        print("self.overrid_dynmaic_commnad:  ", self.overrid_dynmaic_commnad)
        print("self.overrideen_command: ", self.overrideen_command)
         self.elegant_path = "/Users/anwar/Downloads/sdds/darwin-x86/"
            self.sdds_path = "/Users/anwar/Downloads/sdds/defns.rpn"
        '''
        
        # Check if the elegant_input is not empty
        if elegant_input:
            
             # Extract directory from input_beamline_file
            input_dir = os.path.dirname(self.input_file_path)
            elegant_input_path = os.path.join(input_dir, self.elegant_input_filename)
            
            '''print("DEBUG: Input file path:", self.input_file_path)
            print("DEBUG: Extracted directory:", input_dir)
            print("DEBUG: Elegant input will be written to:", elegant_input_path)'''
            
            # Create the elegant input file and write the content
            with open(elegant_input_path, "w") as file:
                file.write(elegant_input)
            # If not empty, run elegant simulation
            if self.overrid_dynmaic_commnad == True:
                command= f"{self.overrideen_command} {self.input_beam_file}.ele"
            else: 
                if self.os_type == "linux":
                    #command = f"mpirun -np 24 Pelegant {self.input_beam_file}.ele"
                    command= f"{self.elegantPath}elegant -rpnDefns={self.sddsPath} {self.input_beam_file}.ele"
                elif self.os_type == "darwin":
                    #command = f"{self.elegantPath}elegant -rpnDefns={self.sddsPath} track_3b.ele"
                    command = f"{self.elegantPath}elegant -rpnDefns={self.sddsPath} {self.input_beam_file}.ele"
                    #print(command)
                elif self.os_type == "windows":
                    command = f"Pelegant {self.input_beam_file}.ele"
                else:
                    raise RuntimeError(f"Unsupported operating system: {self.os_type}")
            '''
            print("The Command: ", command)
            print("#############$$$$$$$$$$$$$$$$$$$$$ DEBUG $$$$$$$$$$$$$$##############")
            '''

            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            #Check if the simulation was successful
            if result.returncode == 0:
                return elegant_input, True, self.dict_variables
            else:
                print("Error: Elegant simulation failed.")
                return None, False, None
        else:
            # If empty, return an error message
            print("Error: Elegant input file is empty.")
            return None, False, None

    def get_num_particles(self):
        number= 0 if self.num_particles is None else self.num_particles
        return number

    def _check_files_created_successfully(self):
        """Check if files are created successfully."""
        files = subprocess.run(["ls", "-l", self.results_path], stdout=subprocess.PIPE)
        list_of_files = files.stdout.split()
        return len(list_of_files) != 0

    def _get_chroneological_order_elements(self):
        graph = process_lte_file_to_graph(self.output_file, self.beamline_name)
        for node in graph:
            if node['type'] == "WATCH" or node['type'] == "watch":
                self.chronolgical_order_watch_points.append(node['name'])
            if node['type'] == "QUAD" or node['type'] == "quad" or node['type'] == "SBEND" or node['type'] == "spend":
                self.chroneological_order_controllable_vars.append(node['name'])
        return graph
 

    def get_results(self, initialNumParticles = 0):
        # We are at a final iteration of the episode
        if self.itteration <= (self.max_itteration-1):
            if not self._check_files_created_successfully():
                time.sleep(5)
            output_file = self.chronolgical_order_watch_points[self.itteration]
            output_file_path = f"{self.results_path}{output_file}.sdds"
            command = str(self.elegantPath + "sdds2stream " + output_file_path + " -col=x,xp,y,yp,t,p,dt,particleID -pipe=out")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout
            error = result.stderr
            if error:
                self.errors[self.itteration] = error
            data = StringIO(output)
            observation_df = pd.read_csv(data, names=['x', 'xp', 'y', 'yp', 't', 'p', 'dt', 'particleID'], sep='\s+')
            observation_df.to_csv('sdds_output.csv', index=False)
            reward = len(observation_df) 
            self.num_particles = reward
            self.graph = process_lte_file_to_graph(self.output_file, self.beamline_name)
            self.Done = False
            if output_file == "final_WP":
                self.Done = True

            # --- New observation logic ---
            # Save the observation as a CSV for processing
            # !!!!! we don't neccerly need to save the file to .csv then reread it because we already have one that is already read
            csv_path = os.path.join(self.observs_dir, f"obs_{output_file}.csv")
            if not os.path.exists(self.observs_dir):
                os.makedirs(self.observs_dir)
            observation_df[['x', 'y', 'xp', 'yp']].to_csv(csv_path, index=False)
            # Use the new process_particle_data function
            obs_result = process_particle_data(
                file_path=csv_path,
                watch_point=output_file,
                graph=self.graph,
                watch_points=self.chronolgical_order_watch_points,
                n_bins=5,
                initialNumParticles=initialNumParticles
            )
            if obs_result is not None:
                observation = obs_result['nn_representation']
            else:
                observation = np.zeros(55)  # fallback to zeros if processing fails

            self.itteration += 1
            return observation, reward, output_file, self.Done
        else:
            print("ERROR: in get results function")
            print("MOST LIKELY WE ALREADY REACHED THE END OF THE BEAMLINE")
            print("self.itteration", self.itteration)
            print("self.max_itteration", self.max_itteration)
            print("##############")
            return None, 0, None, True
        
    def get_results_for_Scipy(self, initialNumParticles = 0):
        if not self._check_files_created_successfully():
            time.sleep(5)
        output_file= "final_WP"
        output_file_path = f"{self.results_path}{output_file}.sdds"
        command = str(self.elegantPath + "sdds2stream " + output_file_path + " -col=x,xp,y,yp,t,p,dt,particleID -pipe=out")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout
        error = result.stderr
        if error:
            self.errors[self.itteration] = error
        data = StringIO(output)
        observation_df = pd.read_csv(data, names=['x', 'xp', 'y', 'yp', 't', 'p', 'dt', 'particleID'], sep='\s+')
        observation_df.to_csv('sdds_output.csv', index=False)
        reward = len(observation_df) 
        self.num_particles = reward
        self.graph = process_lte_file_to_graph(self.output_file, self.beamline_name)

        self.Done = True

        csv_path = os.path.join(self.observs_dir, f"obs_{output_file}.csv")
        if not os.path.exists(self.observs_dir):
            os.makedirs(self.observs_dir)
        observation_df[['x', 'y', 'xp', 'yp']].to_csv(csv_path, index=False)
        
        return reward, self.Done

    ##############################################
    # Analyze other output files
    # PostProcessing
    def _analyzeBeamlineMagnets(self):
        return self.sdds2df(f'{self.results_path}beamline.mag', columns=['ElementName', 's', 'Profile'])

    # Get and analyze the output file track.sig
    def _analyzeTrackSig(self):
        df = self.sdds2df(self.results_path+self.input_beam_file+'.sig', columns=['ElementName', 's', 'minimum1', 'maximum1', 'minimum3', 'maximum3', 'Sx', 'Sy'])
        df['xmin'] = pd.to_numeric(df['minimum1'], errors='coerce') * 1e3
        df['xmax'] = pd.to_numeric(df['maximum1'], errors='coerce') * 1e3
        df['ymin'] = pd.to_numeric(df['minimum3'], errors='coerce') * 1e3
        df['ymax'] = pd.to_numeric(df['maximum3'], errors='coerce') * 1e3
        df['sigma_x'] = pd.to_numeric(df['Sx'], errors='coerce') * 1e3
        df['sigma_y'] = pd.to_numeric(df['Sy'], errors='coerce') * 1e3
        self.dfTrackSig = df
        return self.dfTrackSig

    # Get and analyze the output file track.cen
    def _analyzeTrackCen(self):
        df = self.sdds2df(self.results_path+self.input_beam_file+'.cen', columns=['ElementName', 's', 'Particles', 'pCentral', 'Cx', 'Cy', 'Charge'])
        df['p'] = 0.511 * pd.to_numeric(df['pCentral'], errors='coerce')  # MeV/c
        df['Cx'] = 1e3 * pd.to_numeric(df['Cx'], errors='coerce')  # mm
        df['Cy'] = 1e3 * pd.to_numeric(df['Cy'], errors='coerce')  # mm
        self.df_cen = df
        return self.df_cen

    # Get and analyze the output file twiss.twi
    def _analyzeTwissSddd(self):
        df = self.sdds2df(f'{self.results_path}twiss.twi', columns=['ElementName', 's', 'betax', 'betay', 'alphax', 'alphay', 'etax', 'etay', 'pCentral0', 'xAperture', 'yAperture'])
        df['xAperture'] = 1e3 * pd.to_numeric(df['xAperture'], errors='coerce')  # mm
        df['yAperture'] = 1e3 * pd.to_numeric(df['yAperture'], errors='coerce')  # mm
        self.df_twi = df
        return self.df_twi

    ##############################################
    def plot_magnets(self):
        dfMagnet = self._analyzeBeamlineMagnets()

        hover = HoverTool(tooltips=[("Name", "@ElementName")])

        '''height=300,  # Increase the height
                width=800,  # Increase the width
                tools=['xbox_zoom', 'xpan', hover],'''
        # Update the opts for larger output
        hv.opts.defaults(
            hv.opts.Curve( 'mag',
                height=200,
                width=800,
                show_grid=True,
                xaxis='bottom',
                yaxis='left',
                show_frame=False,
                show_title=False,
                tools=['xbox_zoom', 'xpan', hover],
                color='black',
                alpha=0.3
            ),
            hv.opts.Curve('fx', 
                show_grid=True,
                xaxis='bottom',
                yaxis='left',
                show_frame=False,
                height=200,
                width=800,
                color='red',
                alpha=0.7, 
                line_width=3),

            hv.opts.Curve('fy',
                show_grid=True,
                xaxis='bottom',
                yaxis='left',
                show_frame=False,
                height=200,
                width=800, 
                color='blue', 
                alpha=0.7, 
                line_width=3),
            hv.opts.Curve('Charge',
                show_grid=True,
                xaxis='bottom',
                yaxis='left',
                show_frame=False,
                show_title=False,
                height=200,
                width=800, 
                color='blue', 
                alpha=0.7, 
                line_width=3),
            hv.opts.Curve('Aper',
                height=200,
                width=800, 
                show_grid=True,
                show_frame=False,
                show_title=False,
                color='gray',
                alpha=0.5, 
                line_width=3)
)

        self.mag = hv.Curve(dfMagnet, kdims=self.dim_s, vdims=['Profile', 'ElementName'], group='mag')
        return self.mag
    
    def _plot_magnet_profile(self):
        df_mag = self.sdds2df(f'{self.results_path}beamline.mag')
        #print(df_mag)

        pio.templates.default = pio.templates["simple_white"]
        fig = go.Figure()

        mag = go.Scatter(
            x=df_mag.s, y=df_mag.Profile, mode='lines', line_width=2, line_color="gray",
            hovertext=df_mag.ElementName, hoverinfo="text", showlegend=False
        )
        return mag

    # Can't be called before running the simulation and self.plot_magnets function
    def plot_sig(self):
        if self.mag == None:
            self.plot_magnets()
        self.dfTrackSig = self._analyzeTrackSig()
        dim_x = hv.Dimension('x', unit='mm', label="x", range=(0, None))
        dim_y = hv.Dimension('y', unit='mm', label="y", range=(0, None))

        # Configure options for Curve.fx and Curve.fy programmatically
        hv.opts.Curve('fx', color='red', alpha=0.7, line_width=3)
        hv.opts.Curve('fy', color='blue', alpha=0.7, line_width=3)

        Sx = hv.Curve((self.dfTrackSig.s, self.dfTrackSig.sigma_x), label='σx', kdims=self.dim_s, vdims=dim_x, group='fx')
        Sy = hv.Curve((self.dfTrackSig.s, self.dfTrackSig.sigma_y), label='σy', kdims=self.dim_s, vdims=dim_y, group='fy')

        return (Sx * Sy + self.mag).cols(1)
        
    # Can't be called before running the simulation and self.plot_magnets function
    def plot_cen(self):
        if self.mag == None:
            self.plot_magnets()
        self.dfTrackCen = self._analyzeTrackCen()

        dim_Q = hv.Dimension('Charge Q', range=(0, None), unit='nC')

        #%opts Curve.Charge (color='blue', alpha=0.7, line_width=3)
        
        Charge = hv.Curve((self.dfTrackCen.s, self.dfTrackCen.Charge * 1e9), label='Charge', kdims=self.dim_s, vdims=dim_Q, group='Charge')

        return (Charge + self.mag).cols(1)

    # Can't be called before running the simulation and self.plot_magnets, plot_cen, and plot_sig functions
    def plot_twiss(self):
        if self.mag ==None:
            self.plot_magnets()
        if self.dfTrackSig is None:
            self.dfTrackSig = self._analyzeTrackSig()
        if self.dfTrackCen is None:
            self.dfTrackCen = self._analyzeTrackCen()

        dfTwissSdds = self._analyzeTwissSddd()
        dfTwissSdds.to_csv("people.csv")

        
        dim_p = hv.Dimension('p', unit='MeV/c', label="p", range=(0, None))

        #%opts Curve.Aper (color='gray', alpha=0.5, line_width=3)


        dim_sigma = hv.Dimension('sigma', unit='mm', label="2σ")
        dim_aper = hv.Dimension('aper', unit='mm', label="Aperture")

        Ax = hv.Curve((dfTwissSdds.s, dfTwissSdds.xAperture), label='Aperture', kdims=self.dim_s, vdims=dim_sigma, group='Aper')
        self.Ax_pos = Ax
        self.Ax_neg = hv.Curve((dfTwissSdds.s, -dfTwissSdds.xAperture), label='Aperture', kdims=self.dim_s, vdims=dim_sigma, group='Aper')

        Cx = hv.Spread((self.dfTrackCen.s, self.dfTrackCen.Cx, 2 * self.dfTrackSig.sigma_x), label='x',
                       kdims=self.dim_s, vdims=[self.dim_x, dim_aper])
        Cy = hv.Spread((self.dfTrackCen.s, self.dfTrackCen.Cy, 2 * self.dfTrackSig.sigma_y), label='y',
                       kdims=self.dim_s, vdims=[self.dim_y, dim_aper])

        return (self.Ax_pos * self.Ax_neg * Cy * Cx + self.mag).cols(1)

    # Can't be called before running the simulation and self.plot_magnets, plot_cen, plot_sig, and plot_twiss functions
    def plot_centroids(self):
        if self.dfTrackCen is None:
            self.dfTrackCen = self._analyzeTrackCen()
        if self.Ax_pos == None:
            self.plot_twiss()
        s_Cx = hv.Curve((self.dfTrackCen.s, self.dfTrackCen.Cx), kdims=self.dim_s, vdims=self.dim_x, label='x', group='fx')
        s_Cy = hv.Curve((self.dfTrackCen.s, self.dfTrackCen.Cy), kdims=self.dim_s, vdims=self.dim_x, label='y' , group='fy')

        return (s_Cx * s_Cy * self.Ax_pos * self.Ax_neg + self.mag).cols(1)

    def visulize(self, values):
        self.run_elegant_simulation(values)
        self.mag = self.plot_magnets()
        sigs = self.plot_sig() 
        #sigs= None
        cents = self.plot_cen()
        twiss = self.plot_twiss()
        centrids = self.plot_centroids()
        return self.mag, sigs, cents, twiss, centrids
    
    def plot_betatron(self):
        """Plot betatron functions (beta_x and beta_y)."""
        df = self.sdds2df(f'{self.results_path}twiss.twi')
        fig = make_subplots(rows=2, shared_xaxes=True,row_heights=[0.80, 0.15]) #

        fig.add_trace(go.Scatter(x=df.s, y=df.betax, mode='lines', line_color="red", name='β<sub>x</sub>'))
        fig.add_trace(go.Scatter(x=df.s, y=df.betay, mode='lines', line_color="blue", name='β<sub>y</sub>'))

        fig.update_yaxes(title_text='β<sub>x,y</sub> (m)', showgrid=True) #(title_text='β<sub>x,y</sub> (m)', showgrid=True, range=(0,50))
        fig.update_xaxes(title_text="s (m)", showgrid=True)
        
        mag = self._plot_magnet_profile()
        fig.add_trace(mag, row=2, col=1)

        fig.update_layout(height=600, hovermode='x unified') 

        
        return fig
    
    def plot_dispersion(self):
        """Plot dispersion functions (D_x and D_y)."""
        df = self.sdds2df(f'{self.results_path}twiss.twi')
        fig = make_subplots(rows=2, shared_xaxes=True, row_heights=[0.8, 0.15])

        fig.add_trace(go.Scatter(x=df.s, y=df.etax, mode='lines', line_color="red", name='D<sub>x</sub>'))
        fig.add_trace(go.Scatter(x=df.s, y=df.etay, mode='lines', line_color="blue", name='D<sub>y</sub>'))

        fig.update_yaxes(title_text='D<sub>x,y</sub> (m)', showgrid=True)
        fig.update_xaxes(title_text="s (m)", showgrid=True)
        
        mag = self._plot_magnet_profile()
        fig.add_trace(mag, row=2, col=1)

        fig.update_layout(height=600, hovermode='x unified')

        return fig
    
    def plot_tune_diagram(self):
        """Generate the tune diagram with resonance lines."""
        twi = f'{self.results_path}twiss.twi'
        # Retrieve and convert nux and nuy to floats
        nux = float(self._sddspar(twi, 'nux'))
        nuy = float(self._sddspar(twi, 'nuy'))

        # Use the converted values in the Latex string
        Latex('ν<sub>x</sub>=%.3f, ν<sub>y</sub>=%.3f' % (nux, nuy))
        diag = f'{self.results_path}resdiag.sdds'
        nux_int = np.floor(nux)
        nuy_int = np.floor(nuy)
        
        command= str(self.elegantPath + "sddsresdiag "+ diag + " -order=8 -integerTunes="+ str(nux_int)+ ","+ str(nuy_int)+ " -superperiodicity=1")
        subprocess.run(command, shell=True)
        #!/Users/anwar/Downloads/sdds/darwin-x86/sddsresdiag $diag -order=8 -integerTunes=$nux_int,$nuy_int -superperiodicity=1
        png = f"{self.results_path}tune_diagram.png"
        #command= str(self.elegantPath + "sddsplot -device=png -output="+ png+ " -col=nux,nuy -graph=line,type=15 "+ diag)
        #subprocess.run(command, shell=True)
        #!/Users/anwar/Downloads/sdds/darwin-x86/sddsplot -device=png -output=$png -col=nux,nuy -graph=line,type=15 $diag
        png = f"{self.results_path}image.png"
        command= str(self.elegantPath + "sddsplot -device=png -output="+ png+ " -split=page -col=nux,nuy -graph=line,type=15 -filter=par,Order,1,4 "+ diag+ " -par=nux,nuy -graph=sym,type=1,subtype=1,scale=3,thick=3 "+ twi)
        subprocess.run(command, shell=True)
        #!/Users/anwar/Downloads/sdds/darwin-x86/sddsplot -device=png -output=$png -split=page -col=nux,nuy -graph=line,type=15 -filter=par,Order,1,4 $diag -par=nux,nuy -graph=sym,type=1,subtype=1,scale=3,thick=3 $twi
        
        return png, Latex('ν<sub>x</sub>=%.3f, ν<sub>y</sub>=%.3f' % (nux, nuy))
    
    def plot_chromaticity(self):
        """Plot chromaticity values (xi_x, xi_y)."""
        xi_x = float(self._sddspar(f'{self.results_path}twiss.twi', 'dnux/dp'))
        xi_y = float(self._sddspar(f'{self.results_path}twiss.twi', 'dnuy/dp'))
        return Latex(r'$\xi_x=%.2f,\ \xi_y=%.2f$' % (xi_x, xi_y))
    
    '''def _sddspar(self, file_name, par_name):
        """Extracts a scalar parameter from an SDDS file."""
        command= str(self.elegantPath + "sdds2stream "+file_name+ " -par="+par_name  )
        s = subprocess.run(command , shell=True, capture_output=True, text=True)
       # s = !/Users/anwar/Downloads/sdds/darwin-x86/sdds2stream $file_name -par=$par_name
        try:
            return float(s[0])
        except ValueError:
            return s[0]'''
        
    def _sddspar(self, file_name, par_name):
        """Retrieve a parameter value from an SDDS file."""
        command = f"{self.elegantPath}sdds2stream {file_name} -par={par_name}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode != 0:
            raise RuntimeError(f"Error running sdds2stream: {result.stderr}")
        
        # Return the output as a string
        return result.stdout.strip()

    def get_s_value(self, w_file):
        """Retrieve the s parameter value from an SDDS file."""
        s_value = self._sddspar(w_file, "s")
        return float(s_value)
    
    def get_mag_3d(self, ele_folder='', Z0=0, X0=0, Y0=0, theta0=0, Element_width=0.3):
        """Generate 3D magnet profile."""
        xyz_file = os.path.join(ele_folder, f"{self.results_path}xyz.sdds")
        mag_file = os.path.join(ele_folder,f"{self.results_path}beamline.mag")
        df_xyz = self.sdds2df(xyz_file)
        df_mag = self.sdds2df(mag_file)

        theta = interpolate.interp1d(df_xyz.s, df_xyz.theta, fill_value=(0, 0), bounds_error=False)
        psi = interpolate.interp1d(df_xyz.s, df_xyz.psi, fill_value=(0, 0), bounds_error=False)
        Xco = interpolate.interp1d(df_xyz.s, df_xyz.X, fill_value=(None, None), bounds_error=False)
        Zco = interpolate.interp1d(df_xyz.s, df_xyz.Z, fill_value=(None, None), bounds_error=False)
        Yco = interpolate.interp1d(df_xyz.s, df_xyz.Y, fill_value=(None, None), bounds_error=False)

        s = df_mag.s.values
        nx = np.cos(theta(s)) * np.sin(psi(s) + np.pi / 2)
        nz = -np.sin(theta(s)) * np.sin(psi(s) + np.pi / 2)
        ny = np.cos(psi(s) + np.pi / 2)

        df_mag['X'] = Xco(s) + Element_width * df_mag['Profile'] * nx
        df_mag['Z'] = Zco(s) + Element_width * df_mag['Profile'] * nz
        df_mag['Y'] = Yco(s) + Element_width * df_mag['Profile'] * ny

        Z_old, X_old = df_mag.Z.values.copy(), df_mag.X.values.copy()
        df_mag['Z'] = Z_old * np.cos(theta0) - X_old * np.sin(theta0)
        df_mag['X'] = Z_old * np.sin(theta0) + X_old * np.cos(theta0)

        df_mag['X'] += X0
        df_mag['Z'] += Z0
        df_mag['Y'] += Y0

        return df_mag
    
    def plot_mag_3d(self, df_mag):
        df=df_mag
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df.Z, y=df.X, mode='lines', line_width=2, line_color="gray",
                hovertext=df.ElementName,
                hoverinfo="text+x+y"
            )
        )

        fig.update_xaxes(title_text="Z (m)")
        fig.update_yaxes(title_text="X (m)", scaleanchor="x", scaleratio=1)
        fig.update_layout(height=600)
        fig.show()

    ''' def process_w(self, input_w, output_w="w.sdds", z0=0):
        output_w=f"{self.results_path}{output_w}"
        """Processes the W file by defining required columns."""
        command = [
            self.elegantPath,"sddsprocess", input_w, output_w,
            "-define=col,x_mm,x 1e3 *,symbol=x,units=mm",
            "-define=col,y_mm,y 1e3 *,symbol=y,units=mm",
            f"-define=col,z_mm,t c_mks * -1 * {z0} + 1e3 *,symbol=z,units=mm",
            "-define=col,xp_mrad,xp 1e3 *,symbol=x',units=mrad",
            "-define=col,yp_mrad,yp 1e3 *,symbol=y',units=mrad",
            "-define=col,E_GeV,p 0.511e-3 *,symbol=E,units=GeV"
        ]
        subprocess.run(command, shell=True)
    def process_w(self, input_w, output_w="w.sdds", z0=0):
        """Processes the W file by defining required columns."""
        output_w = os.path.join(self.results_path, output_w)  # Proper path joining
        
        command = [
            os.path.join(self.elegantPath, "sddsprocess"),  # Full path to executable
            input_w,
            output_w,
            "-define=col,x_mm,x,1e3,*,symbol=x,units=mm",
            "-define=col,y_mm,y,1e3,*,symbol=y,units=mm",
            f"-define=col,z_mm,t,c_mks,*,-1,*,{z0},+,1e3,*,symbol=z,units=mm",
            "-define=col,xp_mrad,xp,1e3,*,symbol=x',units=mrad",
            "-define=col,yp_mrad,yp,1e3,*,symbol=y',units=mrad",
            "-define=col,E_GeV,p,0.511e-3,*,symbol=E,units=GeV"
        ]
        
        # Run without shell=True for better security and reliability
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"sddsprocess failed: {result.stderr}")
        return output_w  # Return the full output path'''
    
    def process_w(self, input_w, output_w="w.sdds", z0=0):
        """Processes the W file by defining required columns."""
        output_w = os.path.join(self.results_path, output_w)
        
        command = [
            os.path.join(self.elegantPath, "sddsprocess"),
            input_w,
            output_w,
            f"-rpndefinitionsfiles={self.sddsPath}",
            "-define=col,x_mm,x 1e3 *,symbol=x,units=mm",
            "-define=col,y_mm,y 1e3 *,symbol=y,units=mm",
            f"-define=col,z_mm,t c_mks * -1 * {z0} + 1e3 *,symbol=z,units=mm",
            "-define=col,xp_mrad,xp 1e3 *,symbol=x',units=mrad",
            "-define=col,yp_mrad,yp 1e3 *,symbol=y',units=mrad",
            "-define=col,E_GeV,p 0.511e-3 *,symbol=E,units=GeV"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"sddsprocess failed: {result.stderr}")
        
        # Verify the file was created
        if not os.path.exists(output_w):
            raise FileNotFoundError(f"Output file not created: {output_w}")
        
        return output_w  # This is critical - must return the path
        
    def get_s_value(self, w_file):
        """Retrieve the s parameter value from an SDDS file."""
        s_value = self._sddspar(w_file,"s")#subprocess.run(["sdds2stream", w_file, "-par=s"], capture_output=True, text=True, shell=True)
        return float(s_value)
                