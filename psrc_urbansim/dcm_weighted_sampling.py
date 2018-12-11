import os
os.sys.path.append(r'D:\udst\urbansim')
os.sys.path.append(r'D:\udst\urbansim\urbansim')
os.sys.path.append(r'D:\udst\urbansim\urbansim\dcm')
from urbansim.models.dcm import network_distance_from_home_to_work
from urbansim.models.dcm import avg_network_distance_from_home_to_work
from urbansim.models.dcm import max_logsum_hbw_am_from_home_to_work_wzone_logsum
from urbansim.models.dcm import logsum_hbw_am_from_home_to_work_wzone_logsum
from urbansim.models.dcm import empden_zone_sector
from urbansim.models.dcm import generalized_cost_from_home_to_work
from urbansim.models.dcm import ln_am_total_transit_time_walk_from_home_to_work

from patsy import dmatrix
from prettytable import PrettyTable
from zbox import toolz as tz

import numpy as np
import pandas as pd
import orca

from urbansim.models import dcm, util
from urbansim.urbanchoice import mnl
from urbansim.utils import misc
from urbansim.utils import yamlio
from urbansim_defaults.utils import yaml_to_class, to_frame, check_nas, _print_number_unplaced, _remove_developed_buildings
import logging
from urbansim.utils.logutil import log_start_finish
import timeit

logger = logging.getLogger(__name__)


def lcm_simulate_sample(cfg, choosers, choosers_filter, buildings, join_tbls, out_fname,
                 supply_fname, vacant_fname,
                 enable_supply_correction=None, cast=False):
    """
    Simulate the location choices for the specified choosers

    Parameters
    ----------
    cfg : string
        The name of the yaml config file from which to read the location
        choice model
    choosers : DataFrameWrapper
        A dataframe of agents doing the choosing
    buildings : DataFrameWrapper
        A dataframe of buildings which the choosers are locating in and which
        have a supply
    join_tbls : list of strings
        A list of land use dataframes to give neighborhood info around the
        buildings - will be joined to the buildings using existing broadcasts.
    out_fname : string
        The column name to write the simulated location to
    supply_fname : string
        The string in the buildings table that indicates the amount of
        available units there are for choosers, vacant or not
    vacant_fname : string
        The string in the buildings table that indicates the amount of vacant
        units there will be for choosers
    enable_supply_correction : Python dict
        Should contain keys "price_col" and "submarket_col" which are set to
        the column names in buildings which contain the column for prices and
        an identifier which segments buildings into submarkets
    cast : boolean
        Should the output be cast to match the existing column.
    """
    cfg = misc.config(cfg)

    choosers_df = to_frame(choosers, [], cfg, additional_columns=[out_fname])

    additional_columns = [supply_fname, vacant_fname]
    if enable_supply_correction is not None and \
            "submarket_col" in enable_supply_correction:
        additional_columns += [enable_supply_correction["submarket_col"]]
    if enable_supply_correction is not None and \
            "price_col" in enable_supply_correction:
        additional_columns += [enable_supply_correction["price_col"]]
    locations_df = to_frame(buildings, join_tbls, cfg,
                            additional_columns=additional_columns)
    #location_df = location_df[location_df[buildings_filter]==1]

    available_units = buildings[supply_fname]
    vacant_units = buildings[vacant_fname]

    print "There are %d total available units" % available_units.sum()
    print "    and %d total choosers" % len(choosers)
    print "    but there are %d overfull buildings" % \
          len(vacant_units[vacant_units < 0])

    vacant_units = vacant_units[vacant_units > 0]

    # sometimes there are vacant units for buildings that are not in the
    # locations_df, which happens for reasons explained in the warning below
    indexes = np.repeat(vacant_units.index.values,
                        vacant_units.values.astype('int'))
    isin = pd.Series(indexes).isin(locations_df.index)
    missing = len(isin[isin == False])
    indexes = indexes[isin.values]
    units = locations_df.loc[indexes].reset_index()
    #units = units[units[buildings_filter]==1]
    check_nas(units)

    print "    for a total of %d temporarily empty units" % vacant_units.sum()
    print "    in %d buildings total in the region" % len(vacant_units)

    if missing > 0:
        print "WARNING: %d indexes aren't found in the locations df -" % \
            missing
        print "    this is usually because of a few records that don't join "
        print "    correctly between the locations df and the aggregations tables"
   
    movers = choosers_df[choosers_df[out_fname] == -1]
    #movers = movers[movers[choosers_filter] == 1]
    print "There are %d total movers for this LCM" % len(movers)

    if len(movers) > vacant_units.sum():
        print "WARNING: Not enough locations for movers"
        print "    reducing locations to size of movers for performance gain"
        movers = movers.head(int(vacant_units.sum()))
    segmented_mnl  = PSRC_SegmentedMNLDiscreteChoiceModel.from_yaml(None, cfg)
    dcm_weighted = MNLDiscreteChoiceModelWeightedSamples(choosers_filter, segmented_mnl)
    for large_area in choosers.residence_large_area2.unique():
        if large_area > -2:
            dcm_weighted.add_weight_column(large_area, 'sample_filter_' + str(int(large_area)))         
    start_time = timeit.default_timer()
    new_units = dcm_weighted.predict(movers, units)
    elapsed = timeit.default_timer() - start_time
    print str(elapsed/60.0)
    #.predict_from_cfg(movers, units, cfg, alternative_ratio = 5)
    # new_units returns nans when there aren't enough units,
    # get rid of them and they'll stay as -1s
    new_units = new_units.dropna()

    # go from units back to buildings
    new_buildings = pd.Series(units.loc[new_units.values][out_fname].values,
                              index=new_units.index)

    choosers.update_col_from_series(out_fname, new_buildings, cast=cast)
    _print_number_unplaced(choosers, out_fname)

    if enable_supply_correction is not None:
        new_prices = buildings[price_col]
        if "clip_final_price_low" in enable_supply_correction:
            new_prices = new_prices.clip(lower=enable_supply_correction[
                "clip_final_price_low"])
        if "clip_final_price_high" in enable_supply_correction:
            new_prices = new_prices.clip(upper=enable_supply_correction[
                "clip_final_price_high"])
        buildings.update_col_from_series(price_col, new_prices)

    vacant_units = buildings[vacant_fname]
    print "    and there are now %d empty units" % vacant_units.sum()
    print "    and %d overfull buildings" % len(vacant_units[vacant_units < 0])

def mnl_interaction_dataset_weighted(choosers, alternatives, SAMPLE_SIZE, choosers_weight_segmentation_col, choosers_seg_value, alteratives_weight_column,  
                            chosenalts=None):
    logger.debug((
        'start: compute MNL interaction dataset with {} choosers, '
        '{} alternatives, and sample_size={}'
        ).format(len(choosers), len(alternatives), SAMPLE_SIZE))
    # filter choosers and their current choices if they point to
    # something that isn't in the alternatives table

    choosers = choosers[choosers[choosers_weight_segmentation_col]==choosers_seg_value]

    if chosenalts is not None:
        isin = chosenalts.isin(alternatives.index)
        try:
            removing = isin.value_counts().loc[False]
        except Exception:
            removing = None
        if removing:
            logger.info((
                "Removing {} choice situations because chosen "
                "alt doesn't exist"
            ).format(removing))
            choosers = choosers[isin]
            chosenalts = chosenalts[isin]

    numchoosers = choosers.shape[0]
    numalts = alternatives.shape[0]

    # TODO: this is currently broken in a situation where
    # SAMPLE_SIZE >= numalts. That may not happen often in
    # practical situations but it should be supported
    # because a) why not? and b) testing.
    alts_idx = np.arange(len(alternatives))
    if SAMPLE_SIZE < numalts:
        # TODO: Use stdlib random.sample to individually choose
        # alternatives for each chooser (to avoid repeatedly choosing the
        # same alternative).
        # random.sample is much faster than np.random.choice.
        #sample = np.random.choice(alts_idx, SAMPLE_SIZE * numchoosers)
        sample = alternatives.sample(SAMPLE_SIZE * numchoosers, weights = alteratives_weight_column, replace = True)
        if chosenalts is not None:
            # replace the first row for each chooser with
            # the currently chosen alternative.
            # chosenalts -> integer position
            sample[::SAMPLE_SIZE] = pd.Series(
                alts_idx, index=alternatives.index).loc[chosenalts].values
    else:
        assert chosenalts is None  # if not sampling, must be simulating
        sample = np.tile(alts_idx, numchoosers)

    if not choosers.index.is_unique:
        raise Exception(
            "ERROR: choosers index is not unique, "
            "sample will not work correctly")
    if not alternatives.index.is_unique:
        raise Exception(
            "ERROR: alternatives index is not unique, "
            "sample will not work correctly")

    #alts_sample = alternatives.take(sample)
    alts_sample = sample
    assert len(alts_sample.index) == SAMPLE_SIZE * len(choosers.index)

    
    alts_sample['join_index'] = np.repeat(choosers.index.values, SAMPLE_SIZE)

    alts_sample = pd.merge(
        alts_sample, choosers, left_on='join_index', right_index=True,
        suffixes=('', '_r'))

    chosen = np.zeros((numchoosers, SAMPLE_SIZE))
    chosen[:, 0] = 1

    logger.debug('finish: compute MNL interaction dataset')

    return sample.index.unique().values, alts_sample, chosen

class  PSRC_MNLDiscreteChoiceModel(dcm.MNLDiscreteChoiceModel):
    @classmethod
    def from_yaml(cls, yaml_str=None, str_or_buffer=None):
        """
        Create a DiscreteChoiceModel instance from a saved YAML configuration.
        Arguments are mutally exclusive.

        Parameters
        ----------
        yaml_str : str, optional
            A YAML string from which to load model.
        str_or_buffer : str or file like, optional
            File name or buffer from which to load YAML.

        Returns
        -------
        MNLDiscreteChoiceModel

        """
        cfg = yamlio.yaml_to_dict(yaml_str, str_or_buffer)

        model = cls(
            cfg['model_expression'],
            cfg['sample_size'],
            probability_mode=cfg.get('probability_mode', 'full_product'),
            choice_mode=cfg.get('choice_mode', 'individual'),
            choosers_fit_filters=cfg.get('choosers_fit_filters', None),
            choosers_predict_filters=cfg.get('choosers_predict_filters', None),
            alts_fit_filters=cfg.get('alts_fit_filters', None),
            alts_predict_filters=cfg.get('alts_predict_filters', None),
            interaction_predict_filters=cfg.get(
                'interaction_predict_filters', None),
            estimation_sample_size=cfg.get('estimation_sample_size', None),
            prediction_sample_size=cfg.get('prediction_sample_size', None),
            choice_column=cfg.get('choice_column', None),
            name=cfg.get('name', None)
        )

        if cfg.get('log_likelihoods', None):
            model.log_likelihoods = cfg['log_likelihoods']
        if cfg.get('fit_parameters', None):
            model.fit_parameters = pd.DataFrame(cfg['fit_parameters'])

        logger.debug('loaded LCM model {} from YAML'.format(model.name))
        return model

    def predict_weighted(self, choosers, alternatives, weights, choosers_weight_segmentation_col, debug=False):
        """
        Choose from among alternatives for a group of agents.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing.
        debug : bool
            If debug is set to true, will set the variable "sim_pdf" on
            the object to store the probabilities for mapping of the
            outcome.

        Returns
        -------
        choices : pandas.Series
            Mapping of chooser ID to alternative ID. Some choosers
            will map to a nan value when there are not enough alternatives
            for all the choosers.

        """
        self.assert_fitted()
        logger.debug('start: predict LCM model {}'.format(self.name))

        choosers, alternatives = self.apply_predict_filters(
            choosers, alternatives)

        if len(choosers) == 0:
            return pd.Series()

        if len(alternatives) == 0:
            return pd.Series(index=choosers.index)

        probabilities = self.probabilities_weighted(
            choosers, alternatives, weights, choosers_weight_segmentation_col, filter_tables=False)

        if debug:
            self.sim_pdf = probabilities

        if self.choice_mode == 'aggregate':
            choices = unit_choice(
                choosers.index.values,
                probabilities.index.get_level_values('alternative_id').values,
                probabilities.values)
        elif self.choice_mode == 'individual':
            def mkchoice(probs):
                probs.reset_index(0, drop=True, inplace=True)
                return np.random.choice(
                    probs.index.values, p=probs.values / probs.sum())
            choices = probabilities.groupby(level='chooser_id', sort=False)\
                .apply(mkchoice)
        else:
            raise ValueError(
                'Unrecognized choice_mode option: {}'.format(self.choice_mode))

        logger.debug('finish: predict LCM model {}'.format(self.name))
        return choices

    def probabilities_weighted(self, choosers, alternatives, weights, choosers_weight_segmentation_col, filter_tables=True):
        """
        Returns the probabilities for a set of choosers to choose
        from among a set of alternatives.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing.
        filter_tables : bool, optional
            If True, filter `choosers` and `alternatives` with prediction
            filters before calculating probabilities.

        Returns
        -------
        probabilities : pandas.Series
            Probability of selection associated with each chooser
            and alternative. Index will be a MultiIndex with alternative
            IDs in the inner index and chooser IDs in the out index.

        """
        logger.debug('start: calculate probabilities for LCM model {}'.format(
            self.name))
        self.assert_fitted()

        if filter_tables:
            choosers, alternatives = self.apply_predict_filters(
                choosers, alternatives)

        if self.prediction_sample_size is not None:
            sample_size = self.prediction_sample_size
        else:
            sample_size = len(alternatives)

        data_set_list = []
        for choosers_seg_value, alternatives_weight_column in weights.iteritems():
            print choosers_seg_value
            if self.probability_mode == 'single_chooser':
                _, merged, _ = mnl_interaction_dataset_weighted(
                    choosers.head(1), alternatives, sample_size)
            elif self.probability_mode == 'full_product':
                _, merged, _ = mnl_interaction_dataset_weighted(
                    choosers, alternatives, sample_size, choosers_weight_segmentation_col, choosers_seg_value, alternatives_weight_column)
            else:
                raise ValueError(
                    'Unrecognized probability_mode option: {}'.format(
                        self.probability_mode))
            data_set_list.append(merged)
        merged =  pd.concat(data_set_list)
        merged = util.apply_filter_query(
            merged, self.interaction_predict_filters)
        model_design = dmatrix(
            self.str_model_expression, data=merged, return_type='dataframe')

        if len(merged) != model_design.as_matrix().shape[0]:
            raise ModelEvaluationError(
                'Simulated data does not have the same length as input.  '
                'This suggests there are null values in one or more of '
                'the input columns.')

        # get the order of the coefficients in the same order as the
        # columns in the design matrix
        coeffs = [self.fit_parameters['Coefficient'][x]
                  for x in model_design.columns]

        # probabilities are returned from mnl_simulate as a 2d array
        # with choosers along rows and alternatives along columns
        if self.probability_mode == 'single_chooser':
            numalts = len(merged)
        else:
            numalts = sample_size


        probabilities = mnl.mnl_simulate(
            model_design.as_matrix(),
            coeffs,
            numalts=numalts, returnprobs=True)

        # want to turn probabilities into a Series with a MultiIndex
        # of chooser IDs and alternative IDs.
        # indexing by chooser ID will get you the probabilities
        # across alternatives for that chooser
        mi = pd.MultiIndex.from_arrays(
            [merged['join_index'].values, merged.index.values],
            names=('chooser_id', 'alternative_id'))
        probabilities = pd.Series(probabilities.flatten(), index=mi)

        logger.debug('finish: calculate probabilities for LCM model {}'.format(
            self.name))
        return probabilities

class PSRC_MNLDiscreteChoiceModelGroup(dcm.MNLDiscreteChoiceModelGroup):
    def predict_weighted(self, choosers, alternatives, weights, choosers_weight_segmentation_col, debug=False):
        """
        Choose from among alternatives for a group of agents after
        segmenting the `choosers` table.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
            Must have a column matching the .segmentation_col attribute.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing.
        debug : bool
            If debug is set to true, will set the variable "sim_pdf" on
            the object to store the probabilities for mapping of the
            outcome.

        Returns
        -------
        choices : pandas.Series
            Mapping of chooser ID to alternative ID. Some choosers
            will map to a nan value when there are not enough alternatives
            for all the choosers.

        """
        logger.debug('start: predict models in LCM group {}'.format(self.name))
        results = []

        for name, df in self._iter_groups(choosers):
            choices = self.models[name].predict_weighted(df, alternatives, weights, choosers_weight_segmentation_col, debug=debug)
            if self.remove_alts and len(alternatives) > 0:
                alternatives = alternatives.loc[
                    ~alternatives.index.isin(choices)]
            results.append(choices)

        logger.debug(
            'finish: predict models in LCM group {}'.format(self.name))
        return pd.concat(results) if results else pd.Series()

class PSRC_SegmentedMNLDiscreteChoiceModel(dcm.SegmentedMNLDiscreteChoiceModel):
    """
    An MNL LCM group that allows segments to have different model expressions
    but otherwise share configurations.

    Parameters
    ----------
    segmentation_col
        Name of column in the choosers table that will be used for groupby.
    sample_size : int
        Number of choices to sample for estimating the model.
    probability_mode : str, optional
        Specify the method to use for calculating probabilities
        during prediction.
        Available string options are 'single_chooser' and 'full_product'.
        In "single chooser" mode one agent is chosen for calculating
        probabilities across all alternatives. In "full product" mode
        probabilities are calculated for every chooser across all alternatives.
        Currently "single chooser" mode must be used with a `choice_mode`
        of 'aggregate' and "full product" mode must be used with a
        `choice_mode` of 'individual'.
    choice_mode : str, optional
        Specify the method to use for making choices among alternatives.
        Available string options are 'individual' and 'aggregate'.
        In "individual" mode choices will be made separately for each chooser.
        In "aggregate" mode choices are made for all choosers at once.
        Aggregate mode implies that an alternative chosen by one agent
        is unavailable to other agents and that the same probabilities
        can be used for all choosers.
        Currently "individual" mode must be used with a `probability_mode`
        of 'full_product' and "aggregate" mode must be used with a
        `probability_mode` of 'single_chooser'.
    choosers_fit_filters : list of str, optional
        Filters applied to choosers table before fitting the model.
    choosers_predict_filters : list of str, optional
        Filters applied to the choosers table before calculating
        new data points.
    alts_fit_filters : list of str, optional
        Filters applied to the alternatives table before fitting the model.
    alts_predict_filters : list of str, optional
        Filters applied to the alternatives table before calculating
        new data points.
    interaction_predict_filters : list of str, optional
        Filters applied to the merged choosers/alternatives table
        before predicting agent choices.
    estimation_sample_size : int, optional
        Whether to sample choosers during estimation
        (needs to be applied after choosers_fit_filters)
    prediction_sample_size : int, optional
        Whether (and how much) to sample alternatives during prediction.
        Note that this can lead to multiple choosers picking the same
        alternative.
    choice_column : optional
        Name of the column in the `alternatives` table that choosers
        should choose. e.g. the 'building_id' column. If not provided
        the alternatives index is used.
    default_model_expr : str, iterable, or dict, optional
        A patsy model expression. Should contain only a right-hand side.
    remove_alts : bool, optional
        Specify how to handle alternatives between prediction for different
        models. If False, the alternatives table is not modified between
        predictions. If True, alternatives that have been chosen
        are removed from the alternatives table before doing another
        round of prediction.
    name : str, optional
        An optional string used to identify the model in places.

    """
    def __init__(
            self, segmentation_col, sample_size,
            probability_mode='full_product', choice_mode='individual',
            choosers_fit_filters=None, choosers_predict_filters=None,
            alts_fit_filters=None, alts_predict_filters=None,
            interaction_predict_filters=None,
            estimation_sample_size=None, prediction_sample_size=None,
            choice_column=None, default_model_expr=None, remove_alts=False,
            name=None):
        self._check_prob_choice_mode_compat(probability_mode, choice_mode)
        self._check_prob_mode_interaction_compat(
            probability_mode, interaction_predict_filters)

        self.segmentation_col = segmentation_col
        self.sample_size = sample_size
        self.probability_mode = probability_mode
        self.choice_mode = choice_mode
        self.choosers_fit_filters = choosers_fit_filters
        self.choosers_predict_filters = choosers_predict_filters
        self.alts_fit_filters = alts_fit_filters
        self.alts_predict_filters = alts_predict_filters
        self.interaction_predict_filters = interaction_predict_filters
        self.estimation_sample_size = estimation_sample_size
        self.prediction_sample_size = prediction_sample_size
        self.choice_column = choice_column
        self.default_model_expr = default_model_expr
        self.remove_alts = remove_alts
        self._group = PSRC_MNLDiscreteChoiceModelGroup(
            segmentation_col, remove_alts=remove_alts)
        self.name = (name if name is not None else
                     'PSRC_SegmentedMNLDiscreteChoiceModel')

    @classmethod
    def from_yaml(cls, yaml_str=None, str_or_buffer=None):
        """
        Create a SegmentedMNLDiscreteChoiceModel instance from a saved YAML
        configuration. Arguments are mutally exclusive.

        Parameters
        ----------
        yaml_str : str, optional
            A YAML string from which to load model.
        str_or_buffer : str or file like, optional
            File name or buffer from which to load YAML.

        Returns
        -------
        SegmentedMNLDiscreteChoiceModel

        """
        cfg = yamlio.yaml_to_dict(yaml_str, str_or_buffer)

        default_model_expr = cfg['default_config']['model_expression']

        seg = cls(
            cfg['segmentation_col'],
            cfg['sample_size'],
            cfg['probability_mode'],
            cfg['choice_mode'],
            cfg['choosers_fit_filters'],
            cfg['choosers_predict_filters'],
            cfg['alts_fit_filters'],
            cfg['alts_predict_filters'],
            cfg['interaction_predict_filters'],
            cfg['estimation_sample_size'],
            cfg['prediction_sample_size'],
            cfg['choice_column'],
            default_model_expr,
            cfg['remove_alts'],
            cfg['name'])

        if "models" not in cfg:
            cfg["models"] = {}

        for name, m in cfg['models'].items():
            m['model_expression'] = m.get(
                'model_expression', default_model_expr)
            m['sample_size'] = cfg['sample_size']
            m['probability_mode'] = cfg['probability_mode']
            m['choice_mode'] = cfg['choice_mode']
            m['choosers_fit_filters'] = None
            m['choosers_predict_filters'] = None
            m['alts_fit_filters'] = None
            m['alts_predict_filters'] = None
            m['interaction_predict_filters'] = \
                cfg['interaction_predict_filters']
            m['estimation_sample_size'] = cfg['estimation_sample_size']
            m['prediction_sample_size'] = cfg['prediction_sample_size']
            m['choice_column'] = cfg['choice_column']

            model = PSRC_MNLDiscreteChoiceModel.from_yaml(
                yamlio.convert_to_yaml(m, None))
            seg._group.add_model(model)

        logger.debug(
            'loaded segmented LCM model {} from YAML'.format(seg.name))
        return seg
    def predict_weighted(self, choosers, alternatives, weights, choosers_weight_segmentation_col, debug=False):
        """
        Choose from among alternatives for a group of agents after
        segmenting the `choosers` table.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
            Must have a column matching the .segmentation_col attribute.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing.
        debug : bool
            If debug is set to true, will set the variable "sim_pdf" on
            the object to store the probabilities for mapping of the
            outcome.

        Returns
        -------
        choices : pandas.Series
            Mapping of chooser ID to alternative ID. Some choosers
            will map to a nan value when there are not enough alternatives
            for all the choosers.

        """
        logger.debug(
            'start: predict models in segmented LCM {}'.format(self.name))
        choosers, alternatives = self._filter_choosers_alts(
            choosers, alternatives)
        #self._group2 = PSRC_MNLDiscreteChoiceModelGroup(self.segmentation_col, remove_alts=self.remove_alts)
        results = self._group.predict_weighted(choosers, alternatives, weights, choosers_weight_segmentation_col, debug=debug)
        logger.debug(
            'finish: predict models in segmented LCM {}'.format(self.name))
        return results

class MNLDiscreteChoiceModelWeightedSamples(dcm.DiscreteChoiceModel):
    """
    Manages a group of discrete choice models that refer to different
    segments of choosers.

    Model names must match the segment names after doing a pandas groupby.

    Parameters
    ----------
    segmentation_col : str
        Name of a column in the table of choosers. Will be used to perform
        a pandas groupby on the choosers table.
    remove_alts : bool, optional
        Specify how to handle alternatives between prediction for different
        models. If False, the alternatives table is not modified between
        predictions. If True, alternatives that have been chosen
        are removed from the alternatives table before doing another
        round of prediction.
    name : str, optional
        A name that may be used in places to identify this group.

    """
    def __init__(self, segmentation_col, model, name=None):
        # alternatives will be sampled for each unique value in this column:
        self.choosers_weight_segmentation_col = segmentation_col
        self.model = model
        self.name = name if name is not None else 'MNLDiscreteChoiceModelWeightedSamples'
        self.weight_columns = {}

    def add_weight_column(self, choosers_value, samples_weight_column):
        """
        Add an MNLDiscreteChoiceModel instance.

        Parameters
        ----------
        model : MNLDiscreteChoiceModel
            Should have a ``.name`` attribute matching one of the segments
            in the choosers table.

        """
        logger.debug(
            'adding weight column {} to LCM weight columns {}'.format(choosers_value, samples_weight_column))
        self.weight_columns[choosers_value] = samples_weight_column

    #def add_model_from_params(
    #        self, name, model_expression, sample_size,
    #        probability_mode='full_product', choice_mode='individual',
    #        choosers_fit_filters=None, choosers_predict_filters=None,
    #        alts_fit_filters=None, alts_predict_filters=None,
    #        interaction_predict_filters=None, estimation_sample_size=None,
    #        prediction_sample_size=None, choice_column=None):
    #    """
    #    Add a model by passing parameters through to MNLDiscreteChoiceModel.

    #    Parameters
    #    ----------
    #    name
    #        Must match a segment in the choosers table.
    #    model_expression : str, iterable, or dict
    #        A patsy model expression. Should contain only a right-hand side.
    #    sample_size : int
    #        Number of choices to sample for estimating the model.
    #    probability_mode : str, optional
    #        Specify the method to use for calculating probabilities
    #        during prediction.
    #        Available string options are 'single_chooser' and 'full_product'.
    #        In "single chooser" mode one agent is chosen for calculating
    #        probabilities across all alternatives. In "full product" mode
    #        probabilities are calculated for every chooser across all
    #        alternatives.
    #    choice_mode : str or callable, optional
    #        Specify the method to use for making choices among alternatives.
    #        Available string options are 'individual' and 'aggregate'.
    #        In "individual" mode choices will be made separately for each
    #        chooser. In "aggregate" mode choices are made for all choosers at
    #        once. Aggregate mode implies that an alternative chosen by one
    #        agent is unavailable to other agents and that the same
    #        probabilities can be used for all choosers.
    #    choosers_fit_filters : list of str, optional
    #        Filters applied to choosers table before fitting the model.
    #    choosers_predict_filters : list of str, optional
    #        Filters applied to the choosers table before calculating
    #        new data points.
    #    alts_fit_filters : list of str, optional
    #        Filters applied to the alternatives table before fitting the model.
    #    alts_predict_filters : list of str, optional
    #        Filters applied to the alternatives table before calculating
    #        new data points.
    #    interaction_predict_filters : list of str, optional
    #        Filters applied to the merged choosers/alternatives table
    #        before predicting agent choices.
    #    estimation_sample_size : int, optional
    #        Whether to sample choosers during estimation
    #        (needs to be applied after choosers_fit_filters)
    #    prediction_sample_size : int, optional
    #        Whether (and how much) to sample alternatives during prediction.
    #        Note that this can lead to multiple choosers picking the same
    #        alternative.
    #    choice_column : optional
    #        Name of the column in the `alternatives` table that choosers
    #        should choose. e.g. the 'building_id' column. If not provided
    #        the alternatives index is used.

    #    """
    #    logger.debug('adding model {} to LCM group {}'.format(name, self.name))
    #    self.models[name] = MNLDiscreteChoiceModel(
    #        model_expression, sample_size,
    #        probability_mode, choice_mode,
    #        choosers_fit_filters, choosers_predict_filters,
    #        alts_fit_filters, alts_predict_filters,
    #        interaction_predict_filters, estimation_sample_size,
    #        prediction_sample_size, choice_column, name)

    def _iter_weight_columns(self, data):
        """
        Iterate over the groups in `data` after grouping by
        `segmentation_col`. Skips any groups for which there
        is no model stored.

        Yields tuples of (name, df) where name is the group key
        and df is the group DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            Must have a column with the same name as `segmentation_col`.

        """
        weight_columns = data.groupby(choosers_weight_segmentation_col )

        for name, group in weight_columns:
            if name not in self.weight_columns:
                continue
            logger.debug(
                'returning weigth column {} in LCM group {}'.format(name, self.name))
            yield name, group

    def apply_fit_filters(self, choosers, alternatives):
        """
        Filter `choosers` and `alternatives` for fitting.
        This is done by filtering each submodel and concatenating
        the results.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing,
            e.g. buildings.

        Returns
        -------
        filtered_choosers, filtered_alts : pandas.DataFrame

        """
        ch = []
        alts = []

        for name, df in self._iter_groups(choosers):
            filtered_choosers, filtered_alts = \
                self.models[name].apply_fit_filters(df, alternatives)
            ch.append(filtered_choosers)
            alts.append(filtered_alts)

        return pd.concat(ch), pd.concat(alts)

    def apply_predict_filters(self, choosers, alternatives):
        """
        Filter `choosers` and `alternatives` for prediction.
        This is done by filtering each submodel and concatenating
        the results.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing,
            e.g. buildings.

        Returns
        -------
        filtered_choosers, filtered_alts : pandas.DataFrame

        """
        ch = []
        alts = []

        for name, df in self._iter_groups(choosers):
            filtered_choosers, filtered_alts = \
                self.models[name].apply_predict_filters(df, alternatives)
            ch.append(filtered_choosers)
            alts.append(filtered_alts)

        filtered_choosers = pd.concat(ch)
        filtered_alts = pd.concat(alts)

        return filtered_choosers, filtered_alts.drop_duplicates()

    def fit(self, choosers, alternatives, current_choice):
        """
        Fit and save models based on given data after segmenting
        the `choosers` table.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
            Must have a column with the same name as the .segmentation_col
            attribute.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing,
            e.g. buildings.
        current_choice
            Name of column in `choosers` that indicates which alternative
            they have currently chosen.

        Returns
        -------
        log_likelihoods : dict of dict
            Keys will be model names and values will be dictionaries of
            log-liklihood values as returned by MNLDiscreteChoiceModel.fit.

        """
        with log_start_finish(
                'fit models in LCM group {}'.format(self.name), logger):
            return {
                name: self.models[name].fit(df, alternatives, current_choice)
                for name, df in self._iter_groups(choosers)}

    @property
    def fitted(self):
        """
        Whether all models in the group have been fitted.

        """
        return (all(m.fitted for m in self.models.values())
                if self.models else False)

    def probabilities(self, choosers, alternatives):
        """
        Returns alternative probabilties for each chooser segment as
        a dictionary keyed by segment name.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
            Must have a column matching the .segmentation_col attribute.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing.

        Returns
        -------
        probabilties : dict of pandas.Series

        """
        logger.debug(
            'start: calculate probabilities in LCM group {}'.format(self.name))
        probs = {}

        for name, df in self._iter_groups(choosers):
            probs[name] = self.models[name].probabilities(df, alternatives)

        logger.debug(
            'finish: calculate probabilities in LCM group {}'.format(
                self.name))
        return probs

    def summed_probabilities(self, choosers, alternatives):
        """
        Returns the sum of probabilities for alternatives across all
        chooser segments.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
            Must have a column matching the .segmentation_col attribute.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing.

        Returns
        -------
        probs : pandas.Series
            Summed probabilities from each segment added together.

        """
        if len(alternatives) == 0 or len(choosers) == 0:
            return pd.Series()

        logger.debug(
            'start: calculate summed probabilities in LCM group {}'.format(
                self.name))
        probs = []

        for name, df in self._iter_groups(choosers):
            probs.append(
                self.models[name].summed_probabilities(df, alternatives))

        add = tz.curry(pd.Series.add, fill_value=0)
        probs = tz.reduce(add, probs)

        logger.debug(
            'finish: calculate summed probabilities in LCM group {}'.format(
                self.name))
        return probs

    def predict(self, choosers, alternatives, debug=False):
        """
        Choose from among alternatives for a group of agents after
        segmenting the `choosers` table.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
            Must have a column matching the .segmentation_col attribute.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing.
        debug : bool
            If debug is set to true, will set the variable "sim_pdf" on
            the object to store the probabilities for mapping of the
            outcome.

        Returns
        -------
        choices : pandas.Series
            Mapping of chooser ID to alternative ID. Some choosers
            will map to a nan value when there are not enough alternatives
            for all the choosers.

        """
        logger.debug('start: predict models in LCM group {}'.format(self.name))
        results = []

        choices = self.model.predict_weighted(choosers, alternatives, self.weight_columns, self.choosers_weight_segmentation_col, debug=debug)
        

        logger.debug(
            'finish: predict models in LCM group {}'.format(self.name))
        return choices

    def choosers_columns_used(self):
        """
        Columns from the choosers table that are used for filtering.

        """
        return list(tz.unique(tz.concat(
            m.choosers_columns_used() for m in self.models.values())))

    def alts_columns_used(self):
        """
        Columns from the alternatives table that are used for filtering.

        """
        return list(tz.unique(tz.concat(
            m.alts_columns_used() for m in self.models.values())))

    def interaction_columns_used(self):
        """
        Columns from the interaction dataset used for filtering and in
        the model. These may come originally from either the choosers or
        alternatives tables.

        """
        return list(tz.unique(tz.concat(
            m.interaction_columns_used() for m in self.models.values())))

    def columns_used(self):
        """
        Columns from any table used in the model. May come from either
        the choosers or alternatives tables.

        """
        return list(tz.unique(tz.concat(
            m.columns_used() for m in self.models.values())))