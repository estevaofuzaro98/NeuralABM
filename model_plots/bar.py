import xarray as xr
from dantro.plot.funcs.generic import make_facet_grid_plot
from dantro.plot.utils import ColorManager
import matplotlib.colors as mcolors
from typing import Union

from utopya.eval import PlotHelper


@make_facet_grid_plot(
    map_as="dataset",
    encodings=("col", "row"),
    supported_hue_styles=("discrete",),
    hue_style="discrete",
    add_guide=False,
    register_as_kind=True,
    overwrite_existing=True,
)
def bar(
    ds: xr.Dataset,
    hlpr: PlotHelper,
    *,
    x: str,
    y: str,
    hue: str = None,
    _is_facetgrid: bool,
    cmap: Union[str, dict, mcolors.Colormap] = None,
    norm: Union[str, dict, mcolors.Normalize] = None,
    vmin: float = None,
    vmax: float = None,
    **plot_kwargs,
):

    if "width" not in plot_kwargs:
        if isinstance(ds[x][0].data.item(), str):
            plot_kwargs["width"] = 1
        else:
            plot_kwargs["width"] = ds[x][1] - ds[x][0]
    if hue:

        # Get vmin and vmax of the color wheel
        if vmin is None:
            vmin = ds[hue].min().item()
        if vmax is None:
            vmax = ds[hue].max().item()

        # Build the colormanager
        cm = ColorManager(
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        cm.norm.vmin = vmin
        cm.norm.vmax = vmax

        plot_kwargs.update(dict(color=[cm.map_to_color(x) for x in ds[hue].data]))

    hlpr.ax.bar(ds[x], ds[y], **plot_kwargs)


@make_facet_grid_plot(
    map_as="dataarray",
    encodings=("col", "row"),
    supported_hue_styles=("discrete",),
    hue_style="discrete",
    add_guide=False,
    register_as_kind=True,
    overwrite_existing=True,
    drop_kwargs=("x", "y"),
)
def hist(
    ds: xr.Dataset,
    hlpr: PlotHelper,
    *,
    _is_facetgrid: bool,
    **plot_kwargs,
):
    hlpr.ax.hist(ds, **plot_kwargs)
