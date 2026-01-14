from typing import Optional, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_interactions import hyperslicer
from mpl_interactions import ipyplot as iplt


class TimeSerieVisualizer:
    """
    A class to visualize time series data from satellite imagery.
    It provides interactive controls to select dates, change bands, and export results.
    """

    def __init__(
        self,
        data: np.ndarray,
        dates: list[str],
        sat_types: Union[list[str], str],
        out_export_csv: str = "./selected_dates.csv",
        out_export_gif: str = "./sen-ts-plot.gif",
        fps: int = 2,
        ax: Optional[plt.Axes] = None,
    ):
        """
        Initializes the TimeSerieVisualizer.

        Parameters:
        - data (np.ndarray): The time series data with shape (T, H, W, C), where:
            - T: Number of time steps.
            - H: Height of the image.
            - W: Width of the image.
            - C: Number of spectral bands.
        - dates (List[str]): List of dates corresponding to each time step.
        - sat_types (Union[List[str], str]): List of satellite types or a single satellite type.
        - out_export_csv (str, optional): Path to save the selected dates as a CSV file. Defaults to "./selected_dates.csv".
        - out_export_gif (str, optional): Path to save the animation as a GIF. Defaults to "./sen-ts-plot.gif".
        - fps (int, optional): Frames per second for the GIF. Defaults to 2.
        - ax (Optional[plt.Axes], optional): Matplotlib axes to use for plotting. If None, a new figure is created.
        """
        # Constants
        FIG_SIZE = (10, 8)
        LUT_DESCRIPTION = {
            0: "Blue (B2 490nm)",
            1: "Green (B3 560nm)",
            2: "Red (B4 665nm)",
            3: "Red-Edge (B5 705nm)",
            4: "Red-Edge2 (B6 470nm)",
            5: "Red-Edge3 (B7 783nm)",
            6: "NIR (B8 842nm)",
            7: "NIR-Red-Edge (B8a 865nm)",
            8: "SWIR (B11 1610nm)",
            9: "SWIR2 (B12 2190nm)",
        }
        FONT = {
            "family": "sans-serif",
            "color": "darkgreen",
            "weight": "normal",
            "size": 12,
        }

        # Attributes
        self.font = FONT
        self.data = data
        self.dates = dates
        self.sat_types = (
            sat_types if isinstance(sat_types, list) else len(dates) * [sat_types]
        )
        self.out_export_csv = out_export_csv
        self.out_export_gif = out_export_gif
        self.fps = fps
        self.selected_dates = []
        self.sen_ctrls, self.tags = None, None
        self.select_button, self.export_button = None, None
        self.prev_button, self.next_button = None, None
        self.bands_tag, self.bands_button = None, None
        self.accordion = None
        self.fig, self.ax = None, ax
        self.ui = None
        self.bands_selected = [2, 0, 1]  # Default bands for RGB
        self.lut_description = LUT_DESCRIPTION
        self.bands = list(self.lut_description.values())
        self.figsize = FIG_SIZE

    def __call__(self):
        """
        Initializes and displays the visualization interface.
        """
        if self.ax is not None:
            self.fig = self.ax.get_figure()
        else:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.setup()
        self.display()

    def display(self):
        """
        Displays the user interface.
        """
        display(self.ui)

    def setup(self):
        """
        Sets up the visualization interface, including controls, buttons, and layout.
        """
        self.setup_fig()
        self.setup_controls()
        self.setup_tags()
        self.setup_buttons()
        self.setup_accordion()
        self.setup_ui()

    def setup_ui(self):
        """
        Configures the layout of the user interface.
        """
        self.panel = widgets.VBox(
            [
                widgets.HBox(
                    [self.prev_button, self.next_button],
                    layout=widgets.Layout(width="fit-content"),
                ),
                widgets.HBox(
                    [self.select_button, self.export_button, self.gif_button],
                    layout=widgets.Layout(width="fit-content"),
                ),
                self.dates_tags,
            ]
        )
        self.ui = widgets.VBox(
            children=[
                self.accordion,
                widgets.VBox(
                    children=[self.sen_ctrls.vbox, self.panel],
                    layout=widgets.Layout(width="fit-content"),
                ),
            ]
        )

    def setup_fig(self):
        """
        Configures the figure and axes for visualization.
        """
        self.fig.canvas.toolbar_visible = True
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.capture_scroll = False

        self.ax.get_xaxis().set_visible(False)  # Hide x-axis
        self.ax.get_yaxis().set_visible(False)  # Hide y-axis

    def setup_controls(self, bands: list[int] = [2, 1, 0]):
        """
        Sets up the controls for visualizing the data.

        Parameters:
        - bands (List[int], optional): List of band indices to use for visualization. Defaults to [2, 1, 0] (RGB).
        """
        self.set_controls(bands)

    def setup_tags(self):
        """
        Sets up the tags for selecting dates and bands.
        """
        self.dates_tags = widgets.TagsInput(
            value=[],
            allowed_tags=self.dates,
            allow_duplicates=False,
        )
        self.bands_tags = widgets.TagsInput(
            value=self.bands[:3][::-1],
            allowed_tags=self.bands,
            allow_duplicates=False,
        )

    def setup_buttons(self):
        """
        Sets up the buttons for interaction (select, export, previous, next, save GIF, change bands).
        """
        self.select_button = widgets.Button(description="Select Date")
        self.select_button.on_click(self._select_date)

        self.export_button = widgets.Button(description="Export")
        self.export_button.on_click(self._export_date)

        self.prev_button = widgets.Button(description="Previous")
        self.prev_button.on_click(self._prev_step)

        self.next_button = widgets.Button(description="Next")
        self.next_button.on_click(self._next_step)

        self.gif_button = widgets.Button(
            description="Save GIF",
            style=dict(
                font_style="italic",
                font_weight="bold",
                font_variant="small-caps",
                text_color="red",
                text_decoration="underline",
            ),
        )
        self.gif_button.on_click(self._save_gif)

        self.bands_button = widgets.Button(
            description="Select Band", button_style="success"
        )
        self.bands_button.on_click(self._change_bands)

    def setup_accordion(self):
        """
        Sets up the accordion widget for band selection.
        """
        self.accordion = widgets.Accordion(
            children=[widgets.VBox([self.bands_tags, self.bands_button])]
        )
        self.accordion.set_title(0, "Bands")

    def prepare_img_to_RGB(
        self, data: np.ndarray, bands: list[int] = [2, 1, 0]
    ) -> np.ndarray:
        """
        Prepares the image data for RGB visualization by selecting and normalizing the specified bands.

        Parameters:
        - data (np.ndarray): Input image data with shape (H, W, C).
        - bands (List[int], optional): List of band indices to use for RGB. Defaults to [2, 1, 0].

        Returns:
        - np.ndarray: RGB image data with shape (H, W, 3).
        """
        return np.clip(data[:, :, :, bands] / 2500, 0, 1)

    def prepare_msk_to_RGB(self, data: np.ndarray) -> np.ndarray:
        """
        Prepares the mask data for RGB visualization using a colormap.

        Parameters:
        - data (np.ndarray): Input mask data with shape (H, W).

        Returns:
        - np.ndarray: RGB mask data with shape (H, W, 3).
        """
        cmap = plt.get_cmap("bone")
        rgba_img = cmap(self.contrast_stretching(data))
        rgb_img = np.delete(rgba_img, 3, axis=3)
        return rgb_img

    def contrast_stretching(self, img: np.ndarray) -> np.ndarray:
        """
        Applies contrast stretching to an image.

        Parameters:
        - img (np.ndarray): Input image data.

        Returns:
        - np.ndarray: Contrast-stretched image data.
        """
        min_val = np.min(img, axis=(0, 1))
        max_val = np.max(img, axis=(0, 1))
        stretched_img = (img - min_val) / (max_val - min_val) * 255
        return stretched_img.astype(np.uint8)

    def transform_data(
        self,
        data: Union[np.ndarray, dict[str, np.ndarray]],
        bands: list[int] = [2, 1, 0],
    ) -> np.ndarray:
        """
        Transforms the input data into RGB format for visualization.

        Parameters:
        - data (Union[np.ndarray, Dict[str, np.ndarray]]): Input data (image, mask, or dictionary of both).
        - bands (List[int], optional): List of band indices to use for RGB. Defaults to [2, 1, 0].

        Returns:
        - np.ndarray: Transformed RGB data.
        """
        if isinstance(data, dict):
            blocks = []
            for k, v in data.items():
                if k in {"img", "pred"}:
                    blocks.append(self.prepare_img_to_RGB(v, bands=bands))
                elif k == "msk":
                    blocks.append(self.prepare_msk_to_RGB(v))
                else:
                    raise NotImplementedError
            return np.concatenate(blocks, axis=2)
        else:
            return self.prepare_img_to_RGB(data, bands=bands)

    def _save_gif(self, *args):
        """
        Saves the current visualization as a GIF.
        """
        self.sen_ctrls.save_animation(
            self.out_export_gif, self.fig, "timeframes", fps=self.fps
        )

    def _datelabel_func(self, timeframes: int) -> str:
        """
        Generates a label for the current timeframe.

        Parameters:
        - timeframes (int): Index of the current timeframe.

        Returns:
        - str: Label containing the date and satellite type.
        """
        return f"Date: {self.dates[timeframes]} - Satellite type: {self.sat_types[timeframes]}"

    def _select_date(self, *args):
        """
        Handles the selection of a date.
        """
        index = self.sen_ctrls.controls["timeframes"].children[0].value
        if self.dates[index] not in self.selected_dates:
            self.selected_dates.append(self.dates[index])
            self.dates_tags.value = self.selected_dates

    def _export_date(self, *args):
        """
        Exports the selected dates to a CSV file.
        """
        df = pd.DataFrame(data={"dates": self.dates_tags.value})
        df.to_csv(self.out_export_csv, index=False)

    def _next_step(self, *args):
        """
        Moves to the next timeframe.
        """
        self.sen_ctrls.controls["timeframes"].children[0].value += 1

    def _prev_step(self, *args):
        """
        Moves to the previous timeframe.
        """
        self.sen_ctrls.controls["timeframes"].children[0].value -= 1

    def set_controls(self, bands_selected: list[int]):
        """
        Sets up the controls for visualizing the data with the selected bands.

        Parameters:
        - bands_selected (List[int]): List of band indices to use for visualization.
        """
        self.sen_ctrls = hyperslicer(
            arr=self.transform_data(self.data, bands_selected),
            play_buttons=True,
            axes=(("timeframes", 0)),
            is_color_image=True,
            display_controls=False,
            controls=self.sen_ctrls,
        )
        with self.sen_ctrls["timeframes"]:
            iplt.title(self._datelabel_func, fontdict=self.font)

    def _change_bands(self, *args):
        """
        Handles the change of bands for visualization.
        """
        if self.bands_tags.value and len(self.bands_tags.value) == 3:
            self.bands_selected = [
                self.bands.index(band_name) for band_name in self.bands_tags.value
            ]
            self.set_controls(self.bands_selected)
        self.display()
