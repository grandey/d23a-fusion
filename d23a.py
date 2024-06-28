"""
d23a:
    Functions that support the analysis contained in the d23a-fusion repository.

Author:
    Benjamin S. Grandey, 2023–2024.
"""

from functools import cache
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from watermark import watermark
import xarray as xr


# Matplotlib settings
sns.set_style('whitegrid')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['axes.titleweight'] = 'bold'  # titles for subplots
plt.rcParams['figure.titleweight'] = 'bold'  # suptitle
plt.rcParams['figure.titlesize'] = 'x-large'  # suptitle
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True  # grid should be behind other elements
plt.rcParams['grid.color'] = '0.95'


# Constants
IN_BASE = Path.cwd() / 'data'  # base directory of input data
WF_COLOR_DICT = {'wf_1e': 'skyblue', 'wf_1f': 'skyblue',  # colours to use when plotting different workflows
                 'wf_2e': 'olive', 'wf_2f': 'olive',
                 'wf_3e': 'darkorange', 'wf_3f': 'darkorange',
                 'wf_4': 'darkred',
                 'lower': '0.5', 'upper': '0.5',
                 'outer': 'darkmagenta', 'effective_0.5': 'hotpink',
                 'mean_1e+2e': 'turquoise', 'mean_1f+2f': 'turquoise',
                 'fusion_1e+2e': 'teal', 'fusion_1f+2f': 'teal',
                 'fusion_2e': 'darkgreen', 'fusion_2f': 'darkgreen', 'fusion_1e': 'darkblue',
                 'triangular_1e+2e': 'slateblue',}
WF_LABEL_DICT = {'wf_1e': 'Workflow 1e ($W_1$)', 'wf_1f': 'Workflow 1f ($W_1$)',  # workflow labels in legends
                 'wf_2e': 'Workflow 2e ($W_2$)', 'wf_2f': 'Workflow 2f ($W_2$)',
                 'wf_3e': 'Workflow 3e ($W_3$)', 'wf_3f': 'Workflow 3f ($W_3$)',
                 'wf_4': 'Workflow 4 ($W_4$)',
                 'lower': 'Lower bound ($L$)', 'upper': 'Upper bound ($U$)',
                 'outer': 'Outer bound ($B$)', 'effective_0.5': 'Effective ($E_{0.5})$',
                 'mean_1e+2e': 'Medium conf. mean ($M$)', 'mean_1f+2f': 'Medium conf. mean ($M$)',
                 'fusion_1e+2e': 'Fusion ($F$)', 'fusion_1f+2f': 'Fusion ($F$)',
                 'fusion_2e': 'Fusion 2e ($F_2$)', 'fusion_2f': 'Fusion 2f ($F_2$)', 'fusion_1e': 'Fusion 1e ($F_1$)',
                 'triangular_1e+2e': 'Triangular fusion ($F\'$)',}
SSP_LABEL_DICT = {'ssp126': 'SSP1-2.6', 'ssp585': 'SSP5-8.5', 'both': 'Across\nscenarios'}
SL_LABEL_DICT = {(False, False): '$\Delta$GMSL, m',  # axis labels etc depend on (rate, bool(gauge)) tuple
                 (False, True): '$\Delta$RSL, m',
                 (True, True): 'RSL rate, mm yr$^{-1}$'}
FIG_DIR = Path.cwd() / 'figs_d23a'  # directory in which to save figures
F_NUM = itertools.count(1)  # main figures counter
S_NUM = itertools.count(1)  # supplementary figures counter
O_NUM = itertools.count(1)  # other figures counter


def get_watermark():
    """Return watermark string, including versions of dependencies."""
    packages = 'matplotlib,numpy,pandas,seaborn,xarray'
    return watermark(machine=True, conda=True, python=True, packages=packages)


@cache
def get_gauge_info(gauge='TANJONG_PAGAR'):
    """
    Get name, ID, latitude, and longitude of tide gauge, using location_list.lst
    (https://doi.org/10.5281/zenodo.6382554).

    Parameters
    ----------
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).

    Returns
    ----------
    gauge_info : dict
        Dictionary containing gauge_name, gauge_id, lat, lon.
    """
    # Read input file into DataFrame
    in_fn = IN_BASE / 'location_list.lst'
    in_df = pd.read_csv(in_fn, sep='\t', names=['gauge_name', 'gauge_id', 'lat', 'lon'])
    # Get data for gauge of interest
    try:
        if type(gauge) == str:
            df = in_df[in_df.gauge_name == gauge]
        else:
            df = in_df[in_df.gauge_id == gauge]
        gauge_info = dict()
        for c in ['gauge_name', 'gauge_id', 'lat', 'lon']:
            gauge_info[c] = df[c].values[0]
    except IndexError:
        raise ValueError(f"gauge='{gauge}' not found.")
    return gauge_info


@cache
def get_fusion_weights(weighting='trapezoidal'):
    """
    Return trapezoidal or triangular weighting function for fusion.

    Parameters
    ----------
    weighting : str
        'trapezoidal' (default) or 'triangular'.

    Returns
    -------
    w_da : xarray DataArray
        DataArray of weights for preferred workflow, with weights depending on probability
    """
    # Get a quantile function corresponding to a projection of total sea level, using default parameters
    w_da = get_sl_qf().copy()  # use as template for w_da, with data to be updated
    # Update data to follow weighting function, with weights depending on probability
    if weighting == 'trapezoidal':  # trapezoid, using 17th and 83rd percentiles
        da1 = w_da.sel(quantiles=slice(0, 0.169999))
        da1[:] = da1.quantiles / 0.17
        da2 = w_da.sel(quantiles=slice(0.17, 0.83))
        da2[:] = 1.
        da3 = w_da.sel(quantiles=slice(0.830001, 1))
        da3[:] = (1 - da3.quantiles) / 0.17
        w_da = xr.concat([da1, da2, da3], dim='quantiles')
    elif weighting == 'triangular':
        w_da.data = 1 - np.abs(w_da.quantiles - 0.5) * 2
    else:
        raise ValueError(f"weighting should be 'trapezoidal' or 'triangular', not '{weighting}'.")
    # Rename
    w_da = w_da.rename('weights')
    return w_da


@cache
def get_sl_qf(workflow='wf_1e', rate=False, scenario='ssp585', year=2100, gauge=None, plot=False):
    """
    Return quantile function corresponding to a projection of total sea-level rise.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e', default), p-box bound ('lower', 'upper', 'outer'),
        effective distribution (e.g. 'effective_0.5'), mean (e.g. 'mean_1e+2e'), or fusion (e.g. 'fusion_1e+2e').
    rate : bool
        If True, return rate of sea-level rise. If False (default), return sea-level rise.
    scenario : str
        Options are 'ssp585' (default), 'ssp126', and 'both' (for p-box bound, mean, or fusion across scenarios).
    year : int
        Year. Default is 2100.
    gauge : int, str, or None.
        ID or name of gauge. If None (default), then use global mean.
    plot : Bool
        Plot the result? Default is False.

    Returns
    -------
    qf_da : xarray DataArray
        DataArray of sea-level rise quantiles in m or mm/yr for different probability levels.
    """
    # Case 1: single workflow, corresponding to one of the alternative projections
    if workflow in ['wf_1e', 'wf_1f', 'wf_2e', 'wf_2f', 'wf_3e', 'wf_3f', 'wf_4']:
        # Find gauge_id for location
        if gauge is None:
            gauge_id = -1
        else:
            gauge_info = get_gauge_info(gauge=gauge)
            gauge_id = gauge_info['gauge_id']
        # Read data
        if rate:
            if gauge is None:  # GMSL rate is not available in ar6.zip
                raise ValueError('rate=True is incompatible with gauge=None.')
            else:  # RSL rate
                in_dir = (IN_BASE / 'ar6-regional-distributions' / 'regional' / 'dist_workflows_rates' / workflow
                          / scenario)
            in_fn = in_dir / 'total-workflow_rates.nc'
            qf_da = xr.open_dataset(in_fn)['sea_level_change_rate'].sel(years=year, locations=gauge_id)
        else:
            if gauge is None:  # GMSL
                in_dir = IN_BASE / 'ar6' / 'global' / 'dist_workflows' / workflow / scenario
            else:  # RSL
                in_dir = IN_BASE / 'ar6-regional-distributions' / 'regional' / 'dist_workflows' / workflow / scenario
            in_fn = in_dir / 'total-workflow.nc'
            qf_da = xr.open_dataset(in_fn)['sea_level_change'].sel(years=year, locations=gauge_id)
        # Change units from mm to m
        if not rate:
            qf_da = qf_da / 1000.
            qf_da.attrs['units'] = 'm'
    # Case 2: lower or upper bound of low-confidence p-box
    elif workflow in ['lower', 'upper']:
        # Contributing workflows (Kopp et al., GMD, 2023)
        if not rate:
            wf_list = ['wf_1e', 'wf_2e', 'wf_3e', 'wf_4']
        else:
            wf_list = ['wf_1f', 'wf_2f', 'wf_3f', 'wf_4']
        # Contributing scenarios
        if scenario == 'both':
            scen_list = ['ssp126', 'ssp585']
        else:
            scen_list = [scenario,]
        # Get quantile function data for each of these workflows and scenarios
        qf_da_list = []
        for wf in wf_list:
            for scen in scen_list:
                qf_da_list.append(get_sl_qf(workflow=wf, rate=rate, scenario=scen, year=year, gauge=gauge))
        concat_da = xr.concat(qf_da_list, 'wf_scen')
        # Find lower or upper bound
        if workflow == 'lower':
            qf_da = concat_da.min(dim='wf_scen')
        else:
            qf_da = concat_da.max(dim='wf_scen')
    # Case 3: Outer bound of p-box
    elif workflow == 'outer':
        # Get data for lower and upper p-box bounds
        lower_da = get_sl_qf(workflow='lower', rate=rate, scenario=scenario, year=year, gauge=gauge)
        upper_da = get_sl_qf(workflow='upper', rate=rate, scenario=scenario, year=year, gauge=gauge)
        # Derive outer bound
        qf_da = xr.concat([lower_da.sel(quantiles=slice(0, 0.5)),  # lower bound below median
                           upper_da.sel(quantiles=slice(0.500001, 1))],  # upper bound above median
                          dim='quantiles')
        med_idx = len(qf_da) // 2  # index corresponding to median
        qf_da[med_idx] = np.nan  # median is undefined
    # Case 4: "effective" quantile function (Rohmer et al., 2019)
    elif 'effective' in workflow:
        # Get data for lower and upper p-box bounds
        lower_da = get_sl_qf(workflow='lower', rate=rate, scenario=scenario, year=year, gauge=gauge)
        upper_da = get_sl_qf(workflow='upper', rate=rate, scenario=scenario, year=year, gauge=gauge)
        # Get constant weight w
        w = float(workflow.split('_')[-1])
        # Derive effective distribution
        qf_da = w * upper_da + (1 - w) * lower_da
    # Case 5: "mean" quantile function
    elif 'mean' in workflow:
        # Contributing scenarios
        if scenario == 'both':
            scen_list = ['ssp126', 'ssp585']
        else:
            scen_list = [scenario,]
        # Get quantile function data for workflows and scenarios
        qf_da_list = []
        for wf in [f'wf_{s}' for s in workflow.split('_')[-1].split('+')]:
            for scen in scen_list:
                qf_da_list.append(get_sl_qf(workflow=wf, rate=rate, scenario=scen, year=year, gauge=gauge))
        concat_da = xr.concat(qf_da_list, dim='wf_scen')
        # Derive mean
        qf_da = concat_da.mean(dim='wf_scen')
    # Case 6: fusion distribution
    elif 'fusion' or 'triangular' in workflow:
        # Get data for preferred workflow and outer bound of p-box
        if '+' in workflow:  # use mean for preferred workflow
            wf = f'mean_{workflow.split("_")[-1]}'
        else:  # use single workflow for preferred workflow
            wf = f'wf_{workflow.split("_")[-1]}'
        pref_da = get_sl_qf(workflow=wf, rate=rate, scenario=scenario, year=year, gauge=gauge)
        outer_da = get_sl_qf(workflow='outer', rate=rate, scenario=scenario, year=year, gauge=gauge)
        # Weighting function, with weights depending on probability p
        if 'triangular' in workflow:
            w_da = get_fusion_weights(weighting='triangular')
        else:
            w_da = get_fusion_weights()  # default is triangular weighting function
        # Derive fusion distribution; rely on automatic broadcasting/alignment
        qf_da = w_da * pref_da + (1 - w_da) * outer_da
        # Correct median (which is currently nan due to nan in outer_da)
        med_idx = len(qf_da) // 2  # index corresponding to median
        qf_da[med_idx] = pref_da[med_idx]  # median follows preferred workflow
    # Plot?
    if plot:
        if 'wf' in workflow:
            linestyle = ':'
        elif 'fusion' in workflow:
            linestyle = '-'
        else:
            linestyle = '--'
        qf_da.plot(y='quantiles', label=workflow, alpha=0.5, linestyle=linestyle)
    # Return result
    return qf_da


@cache
def sample_sl_marginal(workflow='wf_1e', rate=False, scenario='ssp585', year=2100, gauge=None, n_samples=int(1e6),
                       plot=False):
    """
    Sample marginal distribution corresponding to a projection of total sea level.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e', default), p-box bound ('lower', 'upper', 'outer'),
        effective distribution (e.g. 'effective_0.5'), or fusion (e.g. 'fusion_1e+2e').
    rate : bool
        If True, return rate of sea-level rise. If False (default), return sea-level rise.
    scenario : str
        Options are 'ssp126', 'ssp585' (default), and 'both'.
    year : int
        Year. Default is 2100.
    gauge : int, str, or None.
        ID or name of gauge. If None (default), then use global mean.
    n_samples : int
        Number of samples. Default is int(1e6).
    plot : Bool
        Plot diagnostic plots? Default is False.

    Returns
    -------
    marginal_n : numpy array
        A one-dimensional array of randomly drawn samples.
    """
    # Read quantile function data
    qf_da = get_sl_qf(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge, plot=False)
    # Sample uniform distribution
    rng = np.random.default_rng(12345)
    uniform_n = rng.uniform(size=n_samples)
    # For p-box outer bounds, correct values near undefined median
    if workflow == 'outer':
        uniform_n[(uniform_n > 0.49) & (uniform_n <= 0.50)] = 0.49
        uniform_n[(uniform_n > 0.50) & (uniform_n < 0.51)] = 0.51
        qf_da = qf_da.dropna(dim='quantiles')
    # Transform these samples to marginal distribution samples by interpolation of quantile function
    marginal_n = qf_da.interp(quantiles=uniform_n).data
    # Check: are there any NaNs?
    if np.any(np.isnan(marginal_n)):
        print('Warning: NaNs found in marginal_n.')
    # Plot diagnostic plots?
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        sns.ecdfplot(marginal_n, label='marginal_n ECDF', ax=axs[0])
        axs[0].plot(qf_da, qf_da['quantiles'], label='"True" quantile function', linestyle='--')
        sns.histplot(marginal_n, bins=100, label='marginal_n', ax=axs[1])
        for ax in axs:
            ax.legend()
        plt.suptitle(f'{workflow}, rate={rate}, {scenario}, {year}, {gauge}')
        plt.show()
    return marginal_n


def plot_fusion_weights(ax=None):
    """
    Plot weighting function used for fusion.

    Parameters
    ----------
    ax : Axes
        Axes on which to plot. If None (default), then use new axes.

    Returns
    -------
    ax : Axes.
    """
    # Create figure if ax is None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    # Get DataArray containing weighting function
    w_da = get_fusion_weights()
    # Plot shaded regions
    ax.fill_betweenx(w_da.quantiles, 0, w_da, color=WF_COLOR_DICT['mean_1e+2e'], alpha=0.2)
    ax.fill_betweenx(w_da.quantiles, 1, w_da, color=WF_COLOR_DICT['outer'], alpha=0.2)
    # Annotate, label axes etc
    ax.text(0.35, 0.5, WF_LABEL_DICT['mean_1e+2e'],
            fontsize='large', horizontalalignment='center', verticalalignment='center', fontweight='bold')
    for y in [0.94, 0.06]:
        ax.text(0.97, y, WF_LABEL_DICT['outer'],
                fontsize='large', horizontalalignment='right', verticalalignment='center', fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Weight for contribution to fusion')
    ax.set_ylabel('Probability')
    return ax


def plot_sl_qfs(workflows=('wf_1e', 'wf_2e', 'wf_3e', 'wf_4'), bg_workflows=list(), pbox=False,
                rate=False, scenario='ssp585', year=2100, gauge=None, ax=None):
    """
    Plot quantile functions corresponding to projections of total sea level.

    Parameters
    ----------
    workflows : list of str
        List containing AR6 workflows, p-box bounds, effective distributions, and/or fusions.
        Default is ('wf_1e', 'wf_2e', 'wf_3e', 'wf_4').
    bg_workflows : list of str
        List containing workflows to show in lighter colour in background. Default is empty list().
    pbox : bool
        If True, plot p-box. Default is False.
    rate : bool
        If True, return rate of sea-level rise. If False (default), return sea-level rise.
    scenario : str
        Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.
    gauge : int, str, or None.
        ID or name of gauge. If None (default), then use global mean.
    ax : Axes
        Axes on which to plot. If None (default), then use new axes.

    Returns
    -------
    ax : Axes
    """
    # Create figure if ax is None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    # Plot p-box?
    if pbox:
        # Get lower and upper p-box bounds
        lower_da = get_sl_qf(workflow='lower', rate=rate, scenario=scenario, year=year, gauge=gauge)
        upper_da = get_sl_qf(workflow='upper', rate=rate, scenario=scenario, year=year, gauge=gauge)
        # Shade p-box
        ax.fill_betweenx(lower_da.quantiles, lower_da, upper_da, color=WF_COLOR_DICT['outer'], alpha=0.1, label='p-box')
    # Loop over background workflows
    for workflow in bg_workflows:
        # Get quantile function data and plot
        qf_da = get_sl_qf(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        ax.plot(qf_da, qf_da.quantiles, color=WF_COLOR_DICT[workflow], alpha=0.5, linestyle='--',
                label=WF_LABEL_DICT[workflow])
    # Loop over workflows
    for workflow in workflows:
        # Get quantile function data and plot
        qf_da = get_sl_qf(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        ax.plot(qf_da, qf_da.quantiles, color=WF_COLOR_DICT[workflow], alpha=0.9, label=WF_LABEL_DICT[workflow])
    # Customise plot
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability')
    xlabel = SL_LABEL_DICT[(rate, bool(gauge))]
    if scenario == 'both':
        xlabel = xlabel.replace(',', ' across scenarios,')
    else:
        xlabel = xlabel.replace(',', f' under {SSP_LABEL_DICT[scenario]},')
    ax.set_xlabel(xlabel)
    return ax


def plot_sl_marginals(workflows=('wf_1e', 'wf_2e', 'wf_3e', 'wf_4'), bg_workflows=list(),
                      rate=False, scenario='ssp585', year=2100, gauge=None, ax=None):
    """
    Plot marginal densities corresponding to projections of total sea level.

    Parameters
    ----------
    workflows : list of str
        List containing AR6 workflows, p-box bounds, effective distributions, and/or fusions.
        Default is ('wf_1e', 'wf_2e', 'wf_3e', 'wf_4').
    bg_workflows : list of str
        List containing workflows to show in lighter colour in background. Default is empty list().
    rate : bool
        If True, return rate of sea-level rise. If False (default), return sea-level rise.
    scenario : str
        Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.
    gauge : int, str, or None.
        ID or name of gauge. If None (default), then use global mean.
    ax : Axes
        Axes on which to plot. If None (default), then use new axes.

    Returns
    -------
    ax : Axes
    """
    # Create figure if ax is None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    # Loop over background workflows
    for workflow in bg_workflows:
        # Get marginal samples and plot
        marginal_n = sample_sl_marginal(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        sns.kdeplot(marginal_n, color=WF_COLOR_DICT[workflow], cut=0, alpha=0.5, linestyle='--',
                    label=WF_LABEL_DICT[workflow], ax=ax)
    # Loop over workflows
    for workflow in workflows:
        # Get marginal samples and plot
        marginal_n = sample_sl_marginal(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        sns.kdeplot(marginal_n, color=WF_COLOR_DICT[workflow], cut=0, alpha=0.9, label=WF_LABEL_DICT[workflow], ax=ax)
    # Customise plot
    ax.legend(loc='upper right')
    xlabel = SL_LABEL_DICT[(rate, bool(gauge))]
    if scenario == 'both':
        xlabel = xlabel.replace(',', ' across scenarios,')
    else:
        xlabel = xlabel.replace(',', f' under {SSP_LABEL_DICT[scenario]},')
    ax.set_xlabel(xlabel)
    return ax


def plot_sl_violinplot(workflows=('wf_2e', 'fusion_1e+2e', 'outer'),
                       rate=False, scenario='ssp585', year=2100, gauge=None, annotations=True, ax=None):
    """
    Plot violinplot of marginal densities corresponding to projections of total sea level.

    Parameters
    ----------
    workflows : list of str
        List containing AR6 workflows, p-box bounds, effective distributions, and/or fusions.
        Default is ('wf_2e', 'fusion_1e+2e', 'outer').
    rate : bool
        If True, return rate of sea-level rise. If False (default), return sea-level rise.
    scenario : str
        Options are 'ssp126', 'ssp585' (default), and 'both'.
    year : int
        Year. Default is 2100.
    gauge : int, str, or None.
        ID or name of gauge. If None (default), then use global mean.
    annotations : bool
        If True (default), add default annotations.
    ax : Axes
        Axes on which to plot. If None (default), then use new axes.

    Returns
    -------
    ax : Axes
    """
    # Create figure if ax is None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    # Loop over workflows, sample marginal, and save to DataFrame
    samples_df = pd.DataFrame()  # DataFrame to hold samples from different marginals
    for workflow in workflows:
        marginal_n = sample_sl_marginal(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        samples_df[workflow] = marginal_n
    # Violinplot
    sns.violinplot(data=samples_df, cut=0, palette=WF_COLOR_DICT, orient='h', width=0.6, inner=None, ax=ax)
    # Percentiles, based on quantile function
    for i, workflow in enumerate(workflows):
        qf_da = get_sl_qf(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        for p_str, linestyle in {'99th': (0, (1, 4)), '95th': 'dotted', '83rd': 'dashed', '50th': 'dashdot'}.items():
            p = float(p_str[:-2])
            val = qf_da.sel(quantiles=p/100).data  # percentile value
            if i == 0:  # label each percentile only once in legend
                label = p_str
            else:
                label = None
            ax.plot([val, val], [i-0.35, i+0.35], color='0.2', linestyle=linestyle, label=label)
    # Annotations (assuming combination & order of workflows follows default)
    if annotations:
        for p, y, text in zip(
                [50, 99], [0.5, 1.5],
                [f'Centre:\n$F \sim M$',
                 f'Upper tail:\n$F \sim B$']
                ):
            qf_da = get_sl_qf(workflow=workflows[1], rate=rate, scenario=scenario, year=year, gauge=gauge)
            val = qf_da.sel(quantiles=p/100).data
            ax.annotate(text, [val, y], ha='center', va='center', color=WF_COLOR_DICT[workflows[1]])
    # Customise plot
    ax.legend(loc='upper right', title='Percentile', title_fontsize='large')
    ax.set_yticklabels([WF_LABEL_DICT[workflow] for workflow in workflows])
    xlabel = SL_LABEL_DICT[(rate, bool(gauge))]
    if scenario == 'both':
        xlabel = xlabel.replace(',', ' across scenarios,')
    else:
        xlabel = xlabel.replace(',', f' under {SSP_LABEL_DICT[scenario]},')
    ax.set_xlabel(xlabel)
    ax.tick_params(axis='both', labelsize='large')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    if gauge is not None:
        ax.text(1, 1.02, f'Location: {gauge.replace("_", " ").title()}',
                ha='right', va='bottom', fontsize='large', transform=ax.transAxes)
    return ax


def plot_exceedance_heatmap(threshold=1.5, workflows=('lower', 'fusion_1e+2e', 'upper'), rate=False,
                            scenarios=('ssp126', 'ssp585', 'both'), year=2100, gauge=None, ax=None):
    """
    Plot heatmap table showing probability of exceeding a sea-level threshold.

    Parameters
    ----------
    threshold : float
        Threshold to use when calculating probability of exceedance.
    workflows : list of str
        List containing AR6 workflows, p-box bounds, effective distributions, and/or fusions, for table columns.
        Default is ('lower', 'fusion_1e+2e', 'upper').
    rate : bool
        If True, return rate of sea-level rise. If False (default), return sea-level rise.
    scenarios : list str
        List containing scenarios, for table rows. Default is ('ssp126', 'ssp585', 'both').
    year : int
        Year. Default is 2100.
    gauge : int, str, or None.
        ID or name of gauge. If None (default), then use global mean.
    ax : Axes
        Axes on which to plot. If None (default), then use new axes.

    Returns
    -------
    ax : Axes
    """
    # Create figure if ax is None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.5), tight_layout=True)
    # For each combination of workflow and scenario, save probability of exceeding threshold to DataFrame
    p_exceed_df = pd.DataFrame()
    for workflow in workflows:
        for scenario in scenarios:
            marginal_n = sample_sl_marginal(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
            p_exceed = (marginal_n > threshold).mean()
            p_exceed_df.loc[SSP_LABEL_DICT[scenario], WF_LABEL_DICT[workflow]] = p_exceed
    # Plot heatmap
    sns.heatmap(p_exceed_df, annot=True, fmt='.0%', cmap='inferno_r', vmin=0., vmax=1.,
                annot_kws={'weight': 'bold', 'fontsize': 'large'}, ax=ax)
    # Change colorbar labels to percentage
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0., 1.])
    cbar.set_ticklabels(['0%', '100%'])
    # Customise plot
    ax.tick_params(top=False, bottom=False, left=False, right=False, rotation=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    return ax


def plot_percentiles_heatmap(percentiles=('5th', '17th', '50th', '83rd', '95th'),
                             workflows=('wf_1e', 'wf_2e', 'wf_3e', 'wf_4', 'outer', 'effective_0.5', 'mean_1e+2e',
                                        'fusion_1e+2e'),
                             rate=False, scenario='ssp585', year=2100, gauge=None, fmt='.1f', ax=None):
    """
    Plot heatmap table showing percentiles of quantile functions.

    Parameters
    ----------
    percentiles : tuple of str
        List containing percentiles, for table columns. Default is ('5th', '17th', '50th', '83rd', '95th').
    workflows : tuple of str
        List containing workflows etc, for table rows.
        Default is ('wf_1e', 'wf_2e', 'wf_3e', 'wf_4', 'outer', 'effective_0.5', 'mean_1e+2e', 'fusion_1e+2e')
    rate : bool
        If True, return rate of sea-level rise. If False (default), return sea-level rise.
    scenario : str
        Scenario. Default is 'ssp585'.
    year : int
        Year. Default is 2100.
    gauge : int, str, or None.
        ID or name of gauge. If None (default), then use global mean.
    fmt : str.
        Format string to use for values. Default is '.1f'.
    ax : Axes
        Axes on which to plot. If None (default), then use new axes.

    Returns
    -------
    ax : Axes
    """
    # Create figure if ax is None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(len(percentiles), 0.5*len(workflows)), tight_layout=True)
    # For each combination of workflow and percentile, save percentile value to DataFrame
    val_df = pd.DataFrame()
    for workflow in workflows:  # rows
        qf_da = get_sl_qf(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        for perc_str in percentiles:  # columns
            perc_flt = float(perc_str[:-2])  # string -> float (e.g. '50th' -> 50.)
            val = qf_da.sel(quantiles=perc_flt/100).data  # percentile value
            val_df.loc[WF_LABEL_DICT[workflow], perc_str] = val
    # Plot heatmap
    sns.heatmap(val_df, annot=True, fmt=fmt, cmap='plasma_r', cbar=False, annot_kws={'weight': 'bold'}, ax=ax)
    # Customise plot
    ax.grid(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labeltop=True, labelbottom=False, rotation=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    title = f'Percentiles of {SL_LABEL_DICT[(rate, bool(gauge))]}\n'
    if scenario == 'both':
        title = title.replace(',', ' across scenarios,')
    else:
        title = title.replace(',', f' under {SSP_LABEL_DICT[scenario]},')
    ax.set_title(title)
    if gauge is not None:
        ax.text(-0.25, 1.1, f'Location:\n{gauge.replace("_", " ").title()}',
                ha='center', va='bottom', transform=ax.transAxes)
    return ax


def fig_qfs_marginals(workflows_r=(('wf_1e', 'wf_2e', 'wf_3e', 'wf_4'), ('outer', 'effective_0.5'), ('fusion_1e+2e',)),
                      bg_workflows_r=(list(), list(), ('mean_1e+2e', 'outer')),
                      pbox_r=(False, True, False),
                      rate=False, scenario='ssp585', year=2100, gauge=None, xlim=None):
    """
    Composite figure showing sea-level quantile functions (1st col) and marginal densities (2nd col).

    Parameters
    ----------
    workflows_r : list of list of str
        List of lists containing AR6 workflows, p-box bounds, effective distributions, and/or fusions, with each list
        corresponding to a different row of figure (indicated by _r in parameter name).
        Default is (('wf_1e', 'wf_2e', 'wf_3e', 'wf_4'), ('outer', 'effective_0.5'), ('fusion_1e+2e',)).
    bg_workflows_r : list of list of str
        List of lists containing workflows to show in lighter colour in background.
        Default is (list(), list(), ('mean_1e+2e', 'outer')).
    pbox_r : list of bool
        When True, plot p-box. Default is (False, True, False).
    rate : bool
        If True, return rate of sea-level rise. If False (default), return sea-level rise.
    scenario : str
        Options are 'ssp126', 'ssp585' (default), and 'both'.
    year : int
        Year. Default is 2100.
    gauge : int, str, or None.
        ID or name of gauge. If None (default), then use global mean.
    xlim : list or None
        x-axis range. Default is None.

    Returns
    -------
    fig: figure
    axs: array of Axes
    """
    # Create Figure and Axes
    nrows = len(workflows_r)
    fig, axs = plt.subplots(nrows, 2, figsize=(9, 3*nrows+0.3), sharex=False, sharey='col', tight_layout=True)
    # Loop over rows
    for r in range(nrows):
        workflows = workflows_r[r]
        bg_workflows = bg_workflows_r[r]
        pbox = pbox_r[r]
        # 1st column: quantile functions
        try:
            ax = axs[r, 0]
        except IndexError:  # IndexError is encountered if only single row
            ax = axs[0]
        plot_sl_qfs(workflows=workflows, bg_workflows=bg_workflows, pbox=pbox,
                    rate=rate, scenario=scenario, year=year, gauge=gauge, ax=ax)
        # 2nd column: marginal densities
        try:
            ax = axs[r, 1]
        except IndexError:
            ax = axs[1]
        plot_sl_marginals(workflows=workflows, bg_workflows=bg_workflows,
                          rate=rate, scenario=scenario, year=year, gauge=gauge, ax=ax)
    # Customise figure
    for i, ax in enumerate(axs.flatten()):
        ax.set_title(f' ({chr(97+i)})', y=1.0, pad=-4, va='top', loc='left')
        if xlim:
            ax.set_xlim(xlim)
    if gauge is not None:
        ax = axs.flatten()[1]
        ax.text(1, 1.05, f'Location: {gauge.replace("_", " ").title()}',
                ha='right', va='bottom', fontsize='large', transform=ax.transAxes)
    return fig, axs


def name_save_fig(fig,
                  fso='f',  # figure type, either 'f' (main), 's' (supp), or 'o' (other)
                  exts=('pdf', 'png'),  # extension(s) to use
                  close=False):
    """Name & save a figure, and increase counter."""
    # Name based on counter, then update counter (in preparation for next figure)
    if fso == 'f':
        fig_name = f'fig{next(F_NUM):02}'
    elif fso == 's':
        fig_name = f's{next(S_NUM):02}'
    else:
        fig_name = f'o{next(O_NUM):02}'
    # File location based on extension(s)
    for ext in exts:
        # Get constrained layout pads (to preserve values after saving fig)
        w_pad, h_pad, _, _ = fig.get_constrained_layout_pads()
        # Sub-directory
        sub_dir = FIG_DIR.joinpath(f'{fso}_{ext}')
        sub_dir.mkdir(exist_ok=True)
        # Save
        fig_path = sub_dir.joinpath(f'{fig_name}.{ext}')
        fig.savefig(fig_path)
        # Print file name and size
        fig_size = fig_path.stat().st_size / 1024 / 1024  # bytes -> MB
        print(f'Written {fig_name}.{ext} ({fig_size:.2f} MB)')
        # Reset constrained layout pads to previous values
        fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad)
    # Suppress output in notebook?
    if close:
        plt.close()
    return fig_name
