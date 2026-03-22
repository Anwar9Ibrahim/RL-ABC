import subprocess
import tempfile
from PIL import Image
import io
import os
from rl_framework.Utils import run_episode
import holoviews as hv
from bokeh.models import HoverTool
from plotly.subplots import make_subplots
from IPython.display import HTML

class ElegantVisualizer:
    def __init__(self, env, val_dict, vis_dir='vis', seed=None):
        self.val_dict = val_dict
        self.values = list(val_dict.values())
        self.env = env
        self.seed = int(seed if seed is not None else getattr(env, "seed", 0))
        self.vis_dir = f"{vis_dir}_{self.seed}"
        self._check_visulization_folder_is_created()
        run_episode(self.env, source="Values", actions=self.values, memory_episode=None)




    def _check_visulization_folder_is_created(self):
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)   

    def plot_sdds_data(self, w_file, png_path= f"results/image.png"):
        """
        Run sddsplot and return the image as a PIL Image object

        Args:
            w_file (str): Path to the SDDS data file (e.g., 'results/final_WP.sdds')

        Returns:
            PIL.Image: The plotted image
        """
        cmd = [
            f'{self.env.elegantPath}sddsplot',
            '-lay=2,2',
            '-device=lpng',
            f'-output={png_path}',
            '-title=',
            '-col=x,y', '-graph=dot,type=2', w_file, '-endPanel',
            '-col=yp,y', '-graph=dot,type=2', w_file, '-endPanel',
            '-col=x,xp', '-graph=dot,type=2', w_file, '-endPanel',
            '-col=yp,xp', '-graph=dot,type=2', w_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"sddsplot failed: {result.stderr}")

        with open(png_path, 'rb') as f:
            img_data = f.read()
        img = Image.open(io.BytesIO(img_data))
        return img
        
    def plot_sdds_data_edited(self, w_file, png_path='results/image.png'):
        """
        Run sddsplot and return the image as a PIL Image object
        Args:
            w_file (str): Path to the SDDS data file (e.g., 'results/final_WP.sdds')
        Returns:
            PIL.Image: The plotted image
        """
        s = self.env.wrapper.get_s_value(w_file)
        w_file_out = "w.sdds"
        processed_file = self.env.wrapper.process_w(w_file, w_file_out, s)
        cmd = [
            f'{self.env.elegantPath}sddsplot',
            '-lay=2,2',
            '-device=lpng',
            f'-output={png_path}',
            '-title=',
            '-col=x_mm,y_mm', '-graph=dot,type=2', processed_file, '-endPanel',
            '-col=yp_mrad,y_mm', '-graph=dot,type=2', processed_file, '-endPanel',
            '-col=x_mm,xp_mrad', '-graph=dot,type=2', processed_file, '-endPanel',
            '-col=yp_mrad,xp_mrad', '-graph=dot,type=2', processed_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"sddsplot failed: {result.stderr}")
        with open(png_path, 'rb') as f:
            img_data = f.read()
        img = Image.open(io.BytesIO(img_data))
        return img

    def plot_energy_phase_space(self, w_file, png_path="results/energy_phase_space.png"):
        """
        Plot energy vs phase space coordinates
        
        Args:
            w_file (str): Path to the SDDS data file
            png_path (str): Output path for the PNG image
            
        Returns:
            PIL.Image: The plotted image
        """
        # First process the file to get the required columns
        s = self.env.wrapper.get_s_value(w_file)
        processed_file = self.env.wrapper.process_w(w_file, "w_energy.sdds", s)
        
        cmd = [
            f'{self.env.elegantPath}sddsplot',
            '-device=lpng',
            f'-output={png_path}',
            '-lay=2,2',
            '-title=',
            '-col=z_mm,E_GeV', '-graph=dot,type=2', processed_file, '-endPanel',
            '-col=xp_mrad,E_GeV', '-graph=dot,type=2', processed_file, '-endPanel',
            '-col=z_mm,x_mm', '-graph=dot,type=2', processed_file, '-endPanel',
            '-col=xp_mrad,x_mm', '-graph=dot,type=2', processed_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"sddsplot failed: {result.stderr}")

        with open(png_path, 'rb') as f:
            img_data = f.read()
        img = Image.open(io.BytesIO(img_data))
        return img
    
    def plot_magnets(self):
        magnets= self.env.wrapper.plot_magnets()
        hv.extension("bokeh")
        
        hv.save(magnets, f'{self.vis_dir}/magnet_plot.html', fmt='html')

        with open(f'{self.vis_dir}/magnet_plot.html', 'r') as f:
            html_content = f.read()
        return magnets , HTML(html_content)

    def plot_sigs(self):
        sigs= self.env.wrapper.plot_sig()
        hv.extension("bokeh")
        hv.save(sigs, f'{self.vis_dir}/sigs_plot.html', fmt='html')

        '''hv.extension('matplotlib')
        sigs.opts(fig_inches=(60, 30))
        hv.save(sigs, f'{self.vis_dir}/sigs_plot.png', fmt='png')'''
        with open(f'{self.vis_dir}/sigs_plot.html', 'r') as f:
            html_content = f.read()

        return sigs, HTML(html_content)
    
    def plot_cents(self):
        cents= self.env.wrapper.plot_cen()
        hv.extension("bokeh")
        hv.save(cents, f'{self.vis_dir}/cents_plot.html', fmt='html')

        with open(f'{self.vis_dir}/cents_plot.html', 'r') as f:
            html_content = f.read()

        return cents, HTML(html_content)
    
    def plot_twiss(self):
        twiss= self.env.wrapper.plot_twiss()
        hv.extension("bokeh")
        hv.save(twiss, f'{self.vis_dir}/twiss_plot.html', fmt='html')

        with open(f'{self.vis_dir}/twiss_plot.html', 'r') as f:
            html_content = f.read()

        return twiss, HTML(html_content)
    
    def plot_centroids(self):
        centroids= self.env.wrapper.plot_centroids()
        hv.extension("bokeh")
        hv.save(centroids, f'{self.vis_dir}/centroids_plot.html', fmt='html')

        with open(f'{self.vis_dir}/centroids_plot.html', 'r') as f:
            html_content = f.read()

        return centroids, HTML(html_content)
    
    def plot_betatron(self):
        return self.env.wrapper.plot_betatron()
    
    def plot_dispersion(self):
        return self.env.wrapper.plot_dispersion()
    
    def plot_tune_diagram(self):
        return self.env.wrapper.plot_tune_diagram()


    def plot_beam_envelope(self):
        """
        Plot beam envelope (centroid ± 2sigma) with correct rectangular aperture.
        Same as plot_twiss but the aperture uses step interpolation so it appears
        as rectangles instead of triangles.
        """
        w = self.env.wrapper

        # Ensure prerequisite data is loaded (must assign return values,
        # since the internal methods use different attribute names)
        if w.mag is None:
            w.plot_magnets()
        if w.dfTrackSig is None:
            w.dfTrackSig = w._analyzeTrackSig()
        if w.dfTrackCen is None:
            w.dfTrackCen = w._analyzeTrackCen()
        dfTwissSdds = w._analyzeTwissSddd()

        dim_sigma = hv.Dimension('sigma', unit='mm', label="2σ")
        dim_aper = hv.Dimension('aper', unit='mm', label="Aperture")

        # Build step-interpolated aperture data so transitions are vertical
        s_vals = dfTwissSdds.s.values
        x_aper = dfTwissSdds.xAperture.values

        s_step = []
        aper_step_pos = []
        aper_step_neg = []
        for i in range(len(s_vals)):
            if i > 0 and x_aper[i] != x_aper[i - 1]:
                # Insert extra point at the new s with the OLD aperture value
                s_step.append(s_vals[i])
                aper_step_pos.append(x_aper[i - 1])
                aper_step_neg.append(-x_aper[i - 1])
            s_step.append(s_vals[i])
            aper_step_pos.append(x_aper[i])
            aper_step_neg.append(-x_aper[i])

        Ax_pos = hv.Curve((s_step, aper_step_pos), label='Aperture',
                            kdims=w.dim_s, vdims=dim_sigma, group='Aper')
        Ax_neg = hv.Curve((s_step, aper_step_neg), label='Aperture',
                            kdims=w.dim_s, vdims=dim_sigma, group='Aper')

        # Beam envelopes (same as plot_twiss)
        Cx = hv.Spread((w.dfTrackCen.s, w.dfTrackCen.Cx, 2 * w.dfTrackSig.sigma_x),
                        label='x', kdims=w.dim_s, vdims=[w.dim_x, dim_aper])
        Cy = hv.Spread((w.dfTrackCen.s, w.dfTrackCen.Cy, 2 * w.dfTrackSig.sigma_y),
                        label='y', kdims=w.dim_s, vdims=[w.dim_y, dim_aper])

        beam_envelope = (Ax_pos * Ax_neg * Cy * Cx + w.mag).cols(1)

        hv.extension("bokeh")
        hv.save(beam_envelope, f'{self.vis_dir}/beam_envelope_plot.html', fmt='html')

        with open(f'{self.vis_dir}/beam_envelope_plot.html', 'r') as f:
            html_content = f.read()

        return beam_envelope, HTML(html_content)






