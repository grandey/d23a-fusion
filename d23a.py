"""
d23a:
    Functions that support the analysis contained in the d23a-fusion repository.

Author:
    Benjamin S. Grandey, 2023.
"""

from functools import cache
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import warnings
from watermark import watermark
import xarray as xr


# Matplotlib settings
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['axes.titleweight'] = 'bold'  # titles for subplots
plt.rcParams['figure.titleweight'] = 'bold'  # suptitle
plt.rcParams['figure.titlesize'] = 'x-large'  # suptitle
plt.rcParams['grid.color'] = '0.95'


# Constants
IN_BASE = Path.cwd() / 'data'  # base directory of input data
WF_COLOR_DICT = {'wf_1e': 'skyblue', 'wf_1f': 'skyblue',  # colours to use when plotting different workflows
                 'wf_2e': 'olive', 'wf_2f': 'olive',
                 'wf_3e': 'darkorange', 'wf_3f': 'darkorange',
                 'wf_4': 'darkred',
                 'lower': '0.5', 'upper': '0.5',
                 'outer': 'darkmagenta', 'effective_0.5': 'hotpink',
                 'fusion_2e': 'darkgreen', 'fusion_2f': 'darkgreen', 'fusion_1e': 'darkblue'}
FIG_DIR = Path.cwd() / 'figs_d23a'  # directory in which to save figures
F_NUM = itertools.count(1)  # main figures counter
S_NUM = itertools.count(1)  # supplementary figures counter
O_NUM = itertools.count(1)  # other figures counter


def get_watermark():
    """Return watermark string, including versions of dependencies."""
    packages = ('matplotlib,numpy,pandas,seaborn,xarray')
    return watermark(machine=True, conda=True, python=True, packages=packages)


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
def get_rslc_qf(workflow='wf_1e', rate=False, scenario='ssp585', year=2100, gauge='TANJONG_PAGAR', plot=False):
    """
    Return quantile function corresponding to a projection of total RSLC.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e', default), p-box bound ('lower', 'upper', 'outer'),
        effective distribution (e.g. 'effective_0.5'), or fusion (e.g. 'fusion_2e').
    rate : bool
        If True, return RSLC rate. If False (default), return RSLC.
    scenario : str
        Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).
    plot : Bool
        Plot the result? Default is False.

    Returns
    -------
    qf_da : xarray DataArray
        DataArray of RSLC quantiles in m or mm/yr for different probability levels.
    """
    # Find gauge_id for location
    gauge_info = get_gauge_info(gauge=gauge)
    gauge_id = gauge_info['gauge_id']
    # Case 1: single workflow, corresponding to one of the alternative projections
    if workflow in ['wf_1e', 'wf_1f', 'wf_2e', 'wf_2f', 'wf_3e', 'wf_3f', 'wf_4']:
        if not rate:  # RSLC
            in_dir = IN_BASE / 'ar6-regional-distributions' / 'regional' / 'dist_workflows' / workflow / scenario
            in_fn = in_dir / 'total-workflow.nc'
        else:  # RSLC rate
            in_dir = IN_BASE / 'ar6-regional-distributions' / 'regional' / 'dist_workflows_rates' / workflow / scenario
            in_fn = in_dir / 'total-workflow_rates.nc'
        # Does input file exist?
        if not in_fn.exists():
            raise FileNotFoundError(in_fn)
        # Read data
        if not rate:
            qf_da = xr.open_dataset(in_fn)['sea_level_change'].sel(years=year, locations=gauge_id)
        else:
            qf_da = xr.open_dataset(in_fn)['sea_level_change_rate'].sel(years=year, locations=gauge_id)
        # Change units from mm to m
        if not rate:
            qf_da = qf_da / 1000.
            qf_da.attrs['units'] = 'm'
    # Case 2: lower or upper bound of low-confidence p-box
    elif workflow in ['lower', 'upper']:
        # Contributing workflows (https://doi.org/10.5194/egusphere-2023-14)
        if not rate:
            wf_list = ['wf_1e', 'wf_2e', 'wf_3e', 'wf_4']
        else:
            wf_list = ['wf_1f', 'wf_2f', 'wf_3f', 'wf_4']
        # Get quantile function data for each of these workflows
        qf_da_list = []
        for wf in wf_list:
            qf_da_list.append(get_rslc_qf(workflow=wf, rate=rate, scenario=scenario, year=year, gauge=gauge))
        concat_da = xr.concat(qf_da_list, 'wf')
        # Find lower or upper bound
        if workflow == 'lower':
            qf_da = concat_da.min(dim='wf')
        else:
            qf_da = concat_da.max(dim='wf')
    # Case 3: Outer bound of p-box
    elif workflow == 'outer':
        # Get data for lower and upper p-box bounds
        lower_da = get_rslc_qf(workflow='lower', rate=rate, scenario=scenario, year=year, gauge=gauge)
        upper_da = get_rslc_qf(workflow='upper', rate=rate, scenario=scenario, year=year, gauge=gauge)
        # Derive outer bound
        qf_da = xr.concat([lower_da.sel(quantiles=slice(0, 0.5)),  # lower bound below median
                           upper_da.sel(quantiles=slice(0.500001, 1))],  # upper bound above median
                          dim='quantiles')
        med_idx = len(qf_da) // 2  # index corresponding to median
        qf_da[med_idx] = np.nan  # median is undefined
    # Case 4: "effective" quantile function (Rohmer et al., 2019)
    elif 'effective' in workflow:
        # Get data for lower and upper p-box bounds
        lower_da = get_rslc_qf(workflow='lower', rate=rate, scenario=scenario, year=year, gauge=gauge)
        upper_da = get_rslc_qf(workflow='upper', rate=rate, scenario=scenario, year=year, gauge=gauge)
        # Get constant weight w
        w = float(workflow.split('_')[-1])
        # Derive effective distribution
        qf_da = w * upper_da + (1 - w) * lower_da
    # Case 5: fusion distribution
    elif 'fusion' in workflow:
        # Get data for preferred workflow and outer bound of p-box
        wf = f'wf_{workflow.split("_")[-1]}'  # preferred workflow
        pref_da = get_rslc_qf(workflow=wf, rate=rate, scenario=scenario, year=year, gauge=gauge)
        outer_da = get_rslc_qf(workflow='outer', rate=rate, scenario=scenario, year=year, gauge=gauge)
        # Triangular weighting function, with weights depending on probability p
        w_p = 1 - np.abs(pref_da.quantiles - 0.5) * 2
        # Derive fusion distribution; rely on automatic broadcasting/alignment
        qf_da = w_p * pref_da + (1 - w_p) * outer_da
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
def sample_rslc_marginal(workflow='wf_1e', rate=False, scenario='ssp585', year=2100, gauge='TANJONG_PAGAR',
                         n_samples=int(1e6), plot=False):
    """
    Sample marginal distribution corresponding to a projection of total RSLC.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e', default), p-box bound ('lower', 'upper', 'outer'),
        effective distribution (e.g. 'effective_0.5'), or fusion (e.g. 'fusion_2e').
    rate : bool
        If True, return RSLC rate. If False (default), return RSLC.
    scenario : str
        Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).
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
    qf_da = get_rslc_qf(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge, plot=False)
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


def plot_rslc_qfs(workflows=('wf_1e', 'wf_2e', 'wf_3e', 'wf_4'), bg_workflows=list(), pbox=False,
                  rate=False, scenario='ssp585', year=2100, gauge='TANJONG_PAGAR', ax=None):
    """
    Plot quantile functions corresponding to projections of total RSLC.

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
        If True, use RSLC rate. If False (default), use RSLC.
    scenario : str
        Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).
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
        lower_da = get_rslc_qf(workflow='lower', rate=rate, scenario=scenario, year=year, gauge=gauge)
        upper_da = get_rslc_qf(workflow='upper', rate=rate, scenario=scenario, year=year, gauge=gauge)
        # Shade p-box
        ax.fill_betweenx(lower_da.quantiles, lower_da, upper_da, color=WF_COLOR_DICT['outer'], alpha=0.1, label='p-box')
    # Loop over background workflows
    for workflow in bg_workflows:
        # Get quantile function data and plot
        qf_da = get_rslc_qf(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        ax.plot(qf_da, qf_da.quantiles, color=WF_COLOR_DICT[workflow], alpha=0.5, label=workflow, linestyle='--')
    # Loop over workflows
    for workflow in workflows:
        # Get quantile function data and plot
        qf_da = get_rslc_qf(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        ax.plot(qf_da, qf_da.quantiles, color=WF_COLOR_DICT[workflow], alpha=0.9, label=workflow)
    # Customise plot
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability')
    if rate:
        ax.set_xlabel(f'RSLC rate, mm/yr')
    else:
        ax.set_xlabel(f'RSLC, m')
    return ax


def plot_rslc_marginals(workflows=('wf_1e', 'wf_2e', 'wf_3e', 'wf_4'), bg_workflows=list(),
                        rate=False, scenario='ssp585', year=2100, gauge='TANJONG_PAGAR', threshold=None, ax=None):
    """
    Plot marginal distributions corresponding to projections of total RSLC.

    Parameters
    ----------
    workflows : list of str
        List containing AR6 workflows, p-box bounds, effective distributions, and/or fusions.
        Default is ('wf_1e', 'wf_2e', 'wf_3e', 'wf_4').
    bg_workflows : list of str
        List containing workflows to show in lighter colour in background. Default is empty list().
    rate : bool
        If True, use RSLC rate. If False (default), use RSLC.
    scenario : str
        Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).
    threshold : float or None.
        Threshold used for calculating probability of exceedance. Default is None.
    ax : Axes
        Axes on which to plot. If None (default), then use new axes.

    Returns
    -------
    ax : Axes
    """
    # Create figure if ax is None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    # Add line showing threshold
    if threshold:
        ax.axvline(threshold, color='0.5', linestyle=':')
    # Loop over background workflows
    for workflow in bg_workflows:
        # Get marginal samples, calculate probability of exceeding threshold, and plot
        marginal_n = sample_rslc_marginal(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        if threshold:
            p_threshold = (marginal_n > threshold).mean()
            label = f'{workflow} ({p_threshold:.0%})'
        else:
            label = workflow
        sns.kdeplot(marginal_n, color=WF_COLOR_DICT[workflow], cut=0, alpha=0.5, label=label, linestyle='--',  ax=ax)
    # Loop over workflows
    for workflow in workflows:
        # Get marginal samples, calculate probability of exceeding threshold, and plot
        marginal_n = sample_rslc_marginal(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        if threshold:
            p_threshold = (marginal_n > threshold).mean()
            label = f'{workflow} ({p_threshold:.0%})'
        else:
            label = workflow
        sns.kdeplot(marginal_n, color=WF_COLOR_DICT[workflow], cut=0, alpha=0.9, label=label, ax=ax)
    # Customise plot
    ax.legend(loc='upper right')
    if rate:
        ax.set_xlabel(f'RSLC rate, mm/yr')
    else:
        ax.set_xlabel(f'RSLC, m')
    return ax


def plot_rslc_violinplot(workflows=('wf_2e', 'fusion_2e', 'outer'),
                         rate=False, scenario='ssp585', year=2100, gauge='TANJONG_PAGAR', annotations=True, ax=None):
    """
    Plot violinplot of marginal distributions corresponding to projections of total RSLC.

    Parameters
    ----------
    workflows : list of str
        List containing AR6 workflows, p-box bounds, effective distributions, and/or fusions.
        Default is ('wf_2e', 'fusion_2e', 'outer').
    rate : bool
        If True, use RSLC rate. If False (default), use RSLC.
    scenario : str
        Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).
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
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    # Loop over workflows, sample marginal, and save to DataFrame
    samples_df = pd.DataFrame()  # DataFrame to hold samples from different marginals
    for workflow in workflows:
        marginal_n = sample_rslc_marginal(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        samples_df[workflow] = marginal_n
    # Violinplot
    sns.violinplot(data=samples_df, cut=0, palette=WF_COLOR_DICT, orient='h', width=0.6, inner=None, ax=ax)
    # Percentiles, based on quantile function
    for i, workflow in enumerate(workflows):
        qf_da = get_rslc_qf(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
        for p_str, linestyle in {'99.5th': (0, (1, 4)), '95th': 'dotted', '83rd': 'dashed', '50th': 'dashdot'}.items():
            p = float(p_str[:-2])
            val = qf_da.sel(quantiles=p/100).data  # percentile value
            if i == 0:  # label each percentile only once in legend
                label = p_str
            else:
                label = None
            ax.plot([val, val], [i-0.35, i+0.35], color='0.2', linestyle=linestyle, label=label)
    # Annotations (assuming combination & order of workflows follows default)
    if annotations:
        for p, y, text in zip([50, 99.5], [0.5, 1.5],
                              [f'Median:\n{workflows[1]}$\sim${workflows[0]}',
                               f'Upper tail:\n{workflows[1]}$\sim${workflows[2]}']):
            qf_da = get_rslc_qf(workflow=workflows[1], rate=rate, scenario=scenario, year=year, gauge=gauge)
            val = qf_da.sel(quantiles=p/100).data
            ax.annotate(text, [val, y], ha='center', va='center', color=WF_COLOR_DICT[workflows[1]])
    # Customise figure
    ax.legend(loc='upper right', title='Percentile', title_fontsize='large')
    if rate:
        ax.set_xlabel(f'RSLC rate, mm/yr')
    else:
        ax.set_xlabel(f'RSLC, m')
    ax.tick_params(axis='both', labelsize='large')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    return ax


def plot_exceedance_heatmap(threshold=1.5, workflows=('lower', 'fusion_2e', 'upper'), rate=False,
                            scenarios=('ssp126', 'ssp585'), year=2100, gauge='TANJONG_PAGAR', ax=None):
    """
    Plot heatmap table showing probability of exceeding an RSLC threshold.

    Parameters
    ----------
    threshold : float
        Threshold to use when calculating probability of exceedance.
    workflows : list of str
        List containing AR6 workflows, p-box bounds, effective distributions, and/or fusions, for table columns.
        Default is ('lower', 'upper', 'fusion_2e').
    rate : bool
        If True, use RSLC rate. If False (default), use RSLC.
    scenarios : list str
        List containing scenarios, for table rows. Default is ('ssp126', 'ssp585').
    year : int
        Year. Default is 2100.
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).
    ax : Axes
        Axes on which to plot. If None (default), then use new axes.

    Returns
    -------
    ax : Axes
    """
    # Create figure if ax is None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 1), constrained_layout=True)
    # For each combination of workflow and scenario, save probability of exceeding threshold to DataFrame
    p_exceed_df = pd.DataFrame()
    for workflow in workflows:
        for scenario in scenarios:
            marginal_n = sample_rslc_marginal(workflow=workflow, rate=rate, scenario=scenario, year=year, gauge=gauge)
            p_exceed = (marginal_n > threshold).mean()
            p_exceed_df.loc[scenario, workflow] = p_exceed
    # Plot heatmap
    sns.heatmap(p_exceed_df, annot=True, fmt='.0%', cmap='inferno_r', vmin=0., vmax=1., ax=ax)
    return ax


def fig_qfs_marginals(workflows_r=(('wf_1e', 'wf_2e', 'wf_3e', 'wf_4'), ('outer', 'effective_0.5'), ('fusion_2e',)),
                      bg_workflows_r=(list(), list(), ('wf_2e', 'outer')),
                      pbox_r=(False, True, False),
                      rate=False, scenario='ssp585', year=2100, gauge='TANJONG_PAGAR', xlim=None):
    """
    Composite figure showing RSLC quantile functions (1st col) and marginals (2nd col).

    Parameters
    ----------
    workflows_r : list of list of str
        List of lists containing AR6 workflows, p-box bounds, effective distributions, and/or fusions, with each list
        corresponding to a different row of figure (indicated by _r in parameter name).
        Default is (('wf_1e', 'wf_2e', 'wf_3e', 'wf_4'), ('outer', 'effective_0.5'), ('fusion_2e',)).
    bg_workflows_r : list of list of str
        List of lists containing workflows to show in lighter colour in background.
        Default is (list(), list(), ('wf_2e', 'outer')).
    pbox_r : list of bool
        When True, plot p-box. Default is (False, True, False).
    rate : bool
        If True, use RSLC rate. If False (default), use RSLC.
    scenario : str
        Options are 'ssp126' and 'ssp585' (default).
    year : int
        Year. Default is 2100.
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).
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
        plot_rslc_qfs(workflows=workflows, bg_workflows=bg_workflows, pbox=pbox,
                      rate=rate, scenario=scenario, year=year, gauge=gauge, ax=ax)
        # 2nd column: marginals
        try:
            ax = axs[r, 1]
        except IndexError:
            ax = axs[1]
        plot_rslc_marginals(workflows=workflows, bg_workflows=bg_workflows,
                            rate=rate, scenario=scenario, year=year, gauge=gauge, ax=ax)
    # Customise figure
    for i, ax in enumerate(axs.flatten()):
        ax.set_title(f' ({chr(97+i)})', y=1.0, pad=-4, va='top', loc='left')
        if xlim:
            ax.set_xlim(xlim)
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


@cache
def read_sea_level_qf(projection_source='fusion', component='total', scenario='SSP5-8.5', year=2100):
    """
    SUPERSEDED BY get_rslc_qf() ABOVE.

    Read quantile function corresponding to sea-level projection (either AR6 ISMIP6, AR6 SEJ, p-box, or fusion).

    Parameters
    ----------
    projection_source : str
        Projection source. Options are 'ISMIP6'/'model-based' (emulated ISMIP6),
        'SEJ'/'expert-based' (Bamber et al. structured expert judgment),
        'p-box'/'bounding quantile function' (p-box bounding quantile function of ISMIP6 & SEJ),
        and 'fusion' (fusion of ISMIP6 and bounding quantile function, weighted using triangular function; default).
    component : str
        Component of global sea level change. Options are 'GrIS' (Greenland Ice Sheet),
        'EAIS' (East Antarctic Ice Sheet), 'WAIS' (West Antarctic Ice Sheet), and
        'total' (total global-mean sea level; default).
        Note: for ISMIP6, 'PEN' is also included in 'WAIS'.
    scenario : str
        Scenario. Options are 'SSP1-2.6' and 'SSP5-8.5' (default).
    year : int
        Year. Default is 2100.

    Returns
    -------
    qf_da : xarray DataArray
        DataArray of sea level change in metres for different quantiles.
    """
    # Case 1: p-box or fusion
    if projection_source in ['p-box', 'bounding quantile function', 'fusion']:
        # If p-box bounding quantile function...
        if projection_source in ['p-box', 'bounding quantile function']:
            # Call function recursively to get ISMIP6 and SEJ data
            ism_da = read_sea_level_qf(projection_source='ISMIP6', component=component, scenario=scenario, year=year)
            sej_da = read_sea_level_qf(projection_source='SEJ', component=component, scenario=scenario, year=year)
            # Initialize qf_da as copy of ism_da
            qf_da = ism_da.copy()
            # Loop over quantile probabilities
            for q, quantile in enumerate(qf_da.quantiles):
                # Get data for this quantile probability
                ism = ism_da[q].data
                sej = sej_da[q].data
                if quantile == 0.5:  # if median, use mean of ISMIP6 and SEJ
                    qf_da[q] = (ism + sej) / 2
                elif quantile < 0.5:  # if quantile < 0.5, use min
                    qf_da[q] = np.minimum(ism, sej)
                else:  # if > 0.5, use maximum
                    qf_da[q] = np.maximum(ism, sej)
        # If fusion...
        elif projection_source == 'fusion':
            # Call function recursively to get ISMIP6 and p-box bounding quantile function data
            ism_da = read_sea_level_qf(projection_source='ISMIP6', component=component, scenario=scenario, year=year)
            pbox_da = read_sea_level_qf(projection_source='p-box', component=component, scenario=scenario, year=year)
            # Weights for ISMIP6 emulator data: triangular function, with peak at median
            weights_q = 1 - np.abs(ism_da.quantiles - 0.5) * 2  # _q indicates quantile probability dimension
            # Combine ISMIP6 and bounding quantile function data using weights; rely on automatic broadcasting/alignment
            qf_da = weights_q * ism_da + (1 - weights_q) * pbox_da
            # Copy units attribute
            qf_da.attrs['units'] = ism_da.attrs['units']
        # Is result monotonic?
        if np.any((qf_da[1:].data - qf_da[:-1].data) < 0):  # allow difference to equal zero
            warnings.warn(f'read_sea_level_qf{projection_source, component, scenario, year} result not monotonic.')
        # Return result (Case 1)
        return qf_da

    # Case 2: ISMIP6 or SEJ
    # Check projection_source argument and identify corresponding projection_code and workflow_code
    elif projection_source in ['ISMIP6', 'model-based']:  # emulated ISMIP6
        projection_code = 'ismipemu'  # projection_code for individual ice-sheet components
        workflow_code = 'wf_1e'  # workflow_code for total GMSL
    elif projection_source in ['SEJ', 'expert-based']:  # Bamber et al structured expert judgment
        projection_code = 'bamber'
        workflow_code = 'wf_4'
    else:
        raise ValueError(f'Unrecognized argument value: projection_source={projection_source}')
    # Check component argument
    if component not in ['EAIS', 'WAIS', 'PEN', 'GrIS', 'total']:
        raise ValueError(f'Unrecognized argument value: component={component}')
    # Check scenario argument and identify corresponding scenario_code
    if scenario in ['SSP1-2.6', 'SSP5-8.5']:
        scenario_code = scenario.replace('-', '').replace('.', '').lower()
    else:
        raise ValueError(f'Unrecognized argument value: scenario={scenario}')
    # Input directory and file
    if component == 'total':
        in_dir = Path(f'data/ar6/global/dist_workflows/{workflow_code}/{scenario_code}')
        in_fn = in_dir / 'total-workflow.nc'
    else:
        in_dir = Path(f'data/ar6/global/dist_components')
        if component == 'GrIS':
            in_fn = in_dir / f'icesheets-ipccar6-{projection_code}icesheet-{scenario_code}_GIS_globalsl.nc'
        else:
            in_fn = in_dir / f'icesheets-ipccar6-{projection_code}icesheet-{scenario_code}_{component}_globalsl.nc'
    # Does input file exist?
    if not in_fn.exists():
        raise FileNotFoundError(in_fn)
    # Read data
    qf_da = xr.open_dataset(in_fn)['sea_level_change'].squeeze().drop_vars('locations').sel(years=year)
    # Convert units from mm to m
    if qf_da.attrs['units'] == 'mm':
        qf_da *= 1e-3
        qf_da.attrs['units'] = 'm'
    # If ISMIP6 WAIS, also include PEN (assuming perfect dependence)
    if projection_code == 'ismipemu' and component == 'WAIS':
        qf_da += read_sea_level_qf(projection_source='ISMIP6', component='PEN', scenario=scenario, year=year)
        print(f'read_sea_level_qf{projection_source, component, scenario, year}: including PEN in WAIS.')
    # Is result monotonic?
    if np.any((qf_da[1:].data - qf_da[:-1].data) < 0):
        warnings.warn(f'read_sea_level_qf{projection_source, component, scenario, year} result not monotonic.')
    # Return result (Case 2)
    return qf_da


@cache
def sample_sea_level_marginal(projection_source='fusion', component='total', scenario='SSP5-8.5', year=2100,
                              n_samples=int(1e6), plot=False):
    """
    SUPERSEDED BY sample_rslc_marginal() ABOVE.

    sample_sea_level_marginal(projection_source, component, scenario, year, n_samples, plot)

    Sample marginal distribution corresponding to sea-level projection (either AR6 ISMIP6, AR6 SEJ, p-box, or fusion).

    Parameters
    ----------
    projection_source : str
        Projection source. Options are 'ISMIP6'/'model-based' (emulated ISMIP6),
        'SEJ'/'expert-based' (Bamber et al. structured expert judgment),
        'p-box'/'bounding quantile function' (p-box bounding quantile function of ISMIP6 & SEJ),
        and 'fusion' (fusion of ISMIP6 and bounding quantile function, weighted using triangular function; default).
    component : str
        Component of global sea level change. Options are 'GrIS' (Greenland Ice Sheet), 'EAIS' (East Antarctic Ice Sheet),
        'WAIS' (West Antarctic Ice Sheet), and 'total' (total global-mean sea level; default).
        Note: for ISMIP6, 'PEN' is also included in 'WAIS'.
    scenario : str
        Scenario. Options are 'SSP1-2.6' and 'SSP5-8.5' (default).
    year : int
        Year. Default is 2100.
    n_samples : int
        Number of samples. Default is int(1e6).
    plot : bool
        If True, plot diagnostic ECDF and histogram. Default is False.

    Returns
    -------
    marginal_n : numpy array
        A one-dimensional array of randomly drawn samples.
    """
    # Read quantile function data
    qf_da = read_sea_level_qf(projection_source=projection_source, component=component, scenario=scenario, year=year)
    # Sample uniform distribution
    rng = np.random.default_rng(12345)
    uniform_n = rng.uniform(size=n_samples)
    # Transform these samples to marginal distribution samples by interpolation of quantile function
    marginal_n = qf_da.interp(quantiles=uniform_n).data
    # Plot diagnostic plots?
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        sns.ecdfplot(marginal_n, label='marginal_n ECDF', ax=axs[0])
        axs[0].plot(qf_da, qf_da['quantiles'], label='"True" quantile function', linestyle='--')
        sns.histplot(marginal_n, bins=100, label='marginal_n', ax=axs[1])
        for ax in axs:
            ax.legend()
            try:
                ax.set_xlabel(f'{component}, {qf_da.attrs["units"]}')
            except KeyError:
                ax.set_xlabel(component)
        plt.suptitle(f'{projection_source}, {component}, {scenario}, {year}, {n_samples}')
        plt.show()
    return marginal_n
