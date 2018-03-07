from __future__ import print_function, division

import abc
import logging

import numpy as np
import pandas as pd
from patsy import dmatrix
from prettytable import PrettyTable
from zbox import toolz as tz

from urbansim.models import util
from urbansim.exceptions import ModelEvaluationError
from urbansim.urbanchoice import interaction, mnl
from urbansim.utils import yamlio
from urbansim.utils.logutil import log_start_finish

from choicemodels import choicemodels

logger = logging.getLogger(__name__)


class DiscreteChoiceModel(object):
    """
    Abstract base class for discrete choice models.

    """
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def _check_prob_choice_mode_compat(probability_mode, choice_mode):
        """
        Check that the probability and choice modes are compatibly with
        each other. Currently 'single_chooser' must be paired with
        'aggregate' and 'full_product' must be paired with 'individual'.

        """
        if (probability_mode == 'full_product' and
                choice_mode == 'aggregate'):
            raise ValueError(
                "'full_product' probability mode is not compatible with "
                "'aggregate' choice mode")

        if (probability_mode == 'single_chooser' and
                choice_mode == 'individual'):
            raise ValueError(
                "'single_chooser' probability mode is not compatible with "
                "'individual' choice mode")

    @staticmethod
    def _check_prob_mode_interaction_compat(
            probability_mode, interaction_predict_filters):
        """
        The 'full_product' probability mode is currently incompatible with
        post-interaction prediction filters, so make sure we don't have
        both of those.

        """
        if (interaction_predict_filters is not None and
                probability_mode == 'full_product'):
            raise ValueError(
                "interaction filters may not be used in "
                "'full_product' mode")

    @abc.abstractmethod
    def apply_fit_filters(self, choosers):
        choosers = util.apply_filter_query(choosers, self.choosers_fit_filters)
        return choosers

    @abc.abstractmethod
    def apply_predict_filters(self, choosers):
        choosers = util.apply_filter_query(
            choosers, self.choosers_predict_filters)
        return choosers

    @abc.abstractproperty
    def fitted(self):
        pass

    @abc.abstractmethod
    def probabilities(self):
        pass

    @abc.abstractmethod
    def summed_probabilities(self):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def choosers_columns_used(self):
        pass

    @abc.abstractmethod
    def alts_columns_used(self):
        pass

    @abc.abstractmethod
    def interaction_columns_used(self):
        pass

    @abc.abstractmethod
    def columns_used(self):
        pass
class BinaryDiscreteChoiceModel(DiscreteChoiceModel):
    """
    A discrete choice model with the ability to store an estimated
    model and predict new data based on the model.
    Based on multinomial logit.

    Parameters
    ----------
    model_expression : str, iterable, or dict
        A patsy model expression. Should contain only a right-hand side.
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
        (needs to be applied after choosers_fit_filters).
    prediction_sample_size : int, optional
        Whether (and how much) to sample alternatives during prediction.
        Note that this can lead to multiple choosers picking the same
        alternative.
    choice_column : optional
        Name of the column in the `alternatives` table that choosers
        should choose. e.g. the 'building_id' column. If not provided
        the alternatives index is used.
    name : optional
        Optional descriptive name for this model that may be used
        in output.

    """
    def __init__(
            self, model_expression, 
            choosers_fit_filters=None, choosers_predict_filters=None,
            alts_fit_filters=None, alts_predict_filters=None,
            interaction_predict_filters=None,
            choice_column=None, name=None):
        self.model_expression = model_expression
        self.choosers_fit_filters = choosers_fit_filters
        self.choosers_predict_filters = choosers_predict_filters
        self.alts_fit_filters = alts_fit_filters
        self.alts_predict_filters = alts_predict_filters
        self.interaction_predict_filters = interaction_predict_filters
        self.choice_column = choice_column
        self.name = name if name is not None else 'BinaryDiscreteChoiceModel'
        self.sim_pdf = None

        self.log_likelihoods = None
        self.fit_parameters = None

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
            choosers_fit_filters=cfg.get('choosers_fit_filters', None),
            choosers_predict_filters=cfg.get('choosers_predict_filters', None),
            alts_fit_filters=cfg.get('alts_fit_filters', None),
            alts_predict_filters=cfg.get('alts_predict_filters', None),
            interaction_predict_filters=cfg.get(
                'interaction_predict_filters', None),
            choice_column=cfg.get('choice_column', None),
            name=cfg.get('name', None)
        )

        if cfg.get('log_likelihoods', None):
            model.log_likelihoods = cfg['log_likelihoods']
        if cfg.get('fit_parameters', None):
            model.fit_parameters = pd.DataFrame(cfg['fit_parameters'])

        logger.debug('loaded LCM model {} from YAML'.format(model.name))
        return model

    @property
    def str_model_expression(self):
        """
        Model expression as a string suitable for use with patsy/statsmodels.

        """
        return util.str_model_expression(
            self.model_expression, add_constant=False)

    def apply_fit_filters(self, choosers):
        """
        Filter `choosers` and `alternatives` for fitting.

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
        return super(BinaryDiscreteChoiceModel, self).apply_fit_filters(
            choosers)

    def apply_predict_filters(self, choosers):
        """
        Filter `choosers` and `alternatives` for prediction.

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
        return super(BinaryDiscreteChoiceModel, self).apply_predict_filters(
            choosers)

    def fit(self, choosers, current_choice):
        """
        Fit and save model parameters based on given data.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing,
            e.g. buildings.
        current_choice : pandas.Series or any
            A Series describing the `alternatives` currently chosen
            by the `choosers`. Should have an index matching `choosers`
            and values matching the index of `alternatives`.

            If a non-Series is given it should be a column in `choosers`.

        Returns
        -------
        log_likelihoods : dict
            Dict of log-liklihood values describing the quality of the
            model fit. Will have keys 'null', 'convergence', and 'ratio'.

        """
        logger.debug('start: fit LCM model {}'.format(self.name))

        if not isinstance(current_choice, pd.Series):
            current_choice = choosers[current_choice]

        choosers = self.apply_fit_filters(choosers)

        current_choice = current_choice.loc[choosers.index]

        model_design = dmatrix(
            self.str_model_expression, data=choosers, return_type='dataframe')

        if len(choosers) != model_design.as_matrix().shape[0]:
            raise ModelEvaluationError(
                'Estimated data does not have the same length as input.  '
                'This suggests there are null values in one or more of '
                'the input columns.')
        logit = choicemodels.Logit(current_choice, model_design)
        results = logit.fit()

        self.log_likelihoods = {
        'null': float(results.llnull),
        'convergence': float(results.llf),
        'ratio': float(results.llr)
        }

        self.fit_parameters = pd.DataFrame({
        'Coefficient': results.params,
        'Std. Error': results.bse,
        'T-Score': results.params / results.bse})

        self.fit_parameters.index = model_design.columns

        logger.debug('finish: fit LCM model {}'.format(self.name))
        return self.log_likelihoods

    @property
    def fitted(self):
        """
        True if model is ready for prediction.

        """
        return self.fit_parameters is not None

    def assert_fitted(self):
        """
        Raises `RuntimeError` if the model is not ready for prediction.

        """
        if not self.fitted:
            raise RuntimeError('Model has not been fit.')

    def report_fit(self):
        """
        Print a report of the fit results.

        """
        if not self.fitted:
            print('Model not yet fit.')
            return

        print('Null Log-liklihood: {0:.3f}'.format(
            self.log_likelihoods['null']))
        print('Log-liklihood at convergence: {0:.3f}'.format(
            self.log_likelihoods['convergence']))
        print('Log-liklihood Ratio: {0:.3f}\n'.format(
            self.log_likelihoods['ratio']))

        tbl = PrettyTable(
            ['Component', ])
        tbl = PrettyTable()

        tbl.add_column('Component', self.fit_parameters.index.values)
        for col in ('Coefficient', 'Std. Error', 'T-Score'):
            tbl.add_column(col, self.fit_parameters[col].values)

        tbl.align['Component'] = 'l'
        tbl.float_format = '.3'

        print(tbl)

    def probabilities(self, choosers, filter_tables=True):
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

        #if self.prediction_sample_size is not None:
        #    sample_size = self.prediction_sample_size
        #else:
        #    sample_size = len(alternatives)

        #if self.probability_mode == 'single_chooser':
        #    _, merged, _ = interaction.mnl_interaction_dataset(
        #        choosers.head(1), alternatives, sample_size)
        #elif self.probability_mode == 'full_product':
        #    _, merged, _ = interaction.mnl_interaction_dataset(
        #        choosers, alternatives, sample_size)
        #else:
        #    raise ValueError(
        #        'Unrecognized probability_mode option: {}'.format(
        #            self.probability_mode))

        #merged = util.apply_filter_query(
        #    merged, self.interaction_predict_filters)
        model_design = dmatrix(
            self.str_model_expression, data=choosers, return_type='dataframe')

        if len(choosers) != model_design.as_matrix().shape[0]:
            raise ModelEvaluationError(
                'Simulated data does not have the same length as input.  '
                'This suggests there are null values in one or more of '
                'the input columns.')

        # get the order of the coefficients in the same order as the
        # columns in the design matrix
        coeffs = [self.fit_parameters['Coefficient'][x]
                  for x in model_design.columns]

        # Constructor requires and observation column, but since we are not estimating any will do so using constant. 
        logit = choicemodels.Logit(model_design.ix[:,0], model_design)

        # Get the prediction probabilities for each chooser
        return pd.DataFrame(logit.predict(coeffs), columns=['probability'], index=model_design.index)
       
        

        # probabilities are returned from mnl_simulate as a 2d array
        # with choosers along rows and alternatives along columns
        #if self.probability_mode == 'single_chooser':
        #    numalts = len(merged)
        #else:
        #    numalts = sample_size

        #probabilities = mnl.mnl_simulate(
        #    model_design.as_matrix(),
        #    coeffs,
        #    numalts=numalts, returnprobs=True)

        ## want to turn probabilities into a Series with a MultiIndex
        ## of chooser IDs and alternative IDs.
        ## indexing by chooser ID will get you the probabilities
        ## across alternatives for that chooser
        #mi = pd.MultiIndex.from_arrays(
        #    [merged['join_index'].values, merged.index.values],
        #    names=('chooser_id', 'alternative_id'))
        #probabilities = pd.Series(probabilities.flatten(), index=mi)

        #logger.debug('finish: calculate probabilities for LCM model {}'.format(
        #    self.name))
        #return probabilities

    def summed_probabilities(self, choosers, alternatives):
        """
        Calculate total probability associated with each alternative.

        Parameters
        ----------
        choosers : pandas.DataFrame
            Table describing the agents making choices, e.g. households.
        alternatives : pandas.DataFrame
            Table describing the things from which agents are choosing.

        Returns
        -------
        probs : pandas.Series
            Total probability associated with each alternative.

        """
        def normalize(s):
            return s / s.sum()

        choosers, alternatives = self.apply_predict_filters(
            choosers, alternatives)
        probs = self.probabilities(choosers, alternatives, filter_tables=False)

        # groupby the the alternatives ID and sum
        if self.probability_mode == 'single_chooser':
            return (
                normalize(probs) * len(choosers)
                ).reset_index(level=0, drop=True)
        elif self.probability_mode == 'full_product':
            return probs.groupby(level=0).apply(normalize)\
                .groupby(level=1).sum()
        else:
            raise ValueError(
                'Unrecognized probability_mode option: {}'.format(
                    self.probability_mode))

    def predict(self, choosers, debug=False):
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

        choosers = self.apply_predict_filters(choosers)

        if len(choosers) == 0:
            return pd.Series()

        probability_df = self.probabilities(
            choosers, filter_tables=False)

        # Monte carlo:
        probability_df['mc'] = np.random.random(len(probability_df))
  
        # True if probibility > random number. 
        probability_df['choice'] = np.where(probability_df.probability > probability_df.mc, 1, 0)

        return(probability_df.choice, probability_df.probability)

        #if debug:
        #    self.sim_pdf = probabilities

        #if self.choice_mode == 'aggregate':
        #    choices = unit_choice(
        #        choosers.index.values,
        #        probabilities.index.get_level_values('alternative_id').values,
        #        probabilities.values)
        #elif self.choice_mode == 'individual':
        #    def mkchoice(probs):
        #        probs.reset_index(0, drop=True, inplace=True)
        #        return np.random.choice(
        #            probs.index.values, p=probs.values / probs.sum())
        #    choices = probabilities.groupby(level='chooser_id', sort=False)\
        #        .apply(mkchoice)
        #else:
        #    raise ValueError(
        #        'Unrecognized choice_mode option: {}'.format(self.choice_mode))

        #logger.debug('finish: predict LCM model {}'.format(self.name))
        #return choices

    def to_dict(self):
        """
        Return a dict respresentation of an MNLDiscreteChoiceModel
        instance.

        """
        return {
            'model_type': 'binarydiscretechoice',
            'model_expression': self.model_expression,
            'name': self.name,
            'choosers_fit_filters': self.choosers_fit_filters,
            'choosers_predict_filters': self.choosers_predict_filters,
            'alts_fit_filters': self.alts_fit_filters,
            'alts_predict_filters': self.alts_predict_filters,
            'interaction_predict_filters': self.interaction_predict_filters,
            'choice_column': self.choice_column,
            'fitted': self.fitted,
            'log_likelihoods': self.log_likelihoods,
            'fit_parameters': (yamlio.frame_to_yaml_safe(self.fit_parameters)
                               if self.fitted else None)
        }

    def to_yaml(self, str_or_buffer=None):
        """
        Save a model respresentation to YAML.

        Parameters
        ----------
        str_or_buffer : str or file like, optional
            By default a YAML string is returned. If a string is
            given here the YAML will be written to that file.
            If an object with a ``.write`` method is given the
            YAML will be written to that object.

        Returns
        -------
        j : str
            YAML is string if `str_or_buffer` is not given.

        """
        logger.debug('serializing LCM model {} to YAML'.format(self.name))
        #if (not isinstance(self.probability_mode, str) or
        #        not isinstance(self.choice_mode, str)):
        #    raise TypeError(
        #        'Cannot serialize model with non-string probability_mode '
        #        'or choice_mode attributes.')
        return yamlio.convert_to_yaml(self.to_dict(), str_or_buffer)

    def choosers_columns_used(self):
        """
        Columns from the choosers table that are used for filtering.

        """
        return list(tz.unique(tz.concatv(
            util.columns_in_filters(self.choosers_predict_filters),
            util.columns_in_filters(self.choosers_fit_filters))))

    def alts_columns_used(self):
        """
        Columns from the alternatives table that are used for filtering.

        """
        return list(tz.unique(tz.concatv(
            util.columns_in_filters(self.alts_predict_filters),
            util.columns_in_filters(self.alts_fit_filters))))

    def interaction_columns_used(self):
        """
        Columns from the interaction dataset used for filtering and in
        the model. These may come originally from either the choosers or
        alternatives tables.

        """
        return list(tz.unique(tz.concatv(
            util.columns_in_filters(self.interaction_predict_filters),
            util.columns_in_formula(self.model_expression))))

    def columns_used(self):
        """
        Columns from any table used in the model. May come from either
        the choosers or alternatives tables.

        """
        return list(tz.unique(tz.concatv(
            self.choosers_columns_used(),
            self.alts_columns_used(),
            self.interaction_columns_used())))

    @classmethod
    def fit_from_cfg(cls, choosers, chosen_fname, cfgname, outcfgname=None):
        """
        Parameters
        ----------
        choosers : DataFrame
            A dataframe in which rows represent choosers.
        chosen_fname : string
            A string indicating the column in the choosers dataframe which
            gives which alternatives the choosers have chosen.
        alternatives : DataFrame
            A table of alternatives. It should include the choices
            from the choosers table as well as other alternatives from
            which to sample.  Values in choosers[chosen_fname] should index
            into the alternatives dataframe.
        cfgname : string
            The name of the yaml config file from which to read the discrete
            choice model.
        outcfgname : string, optional (default cfgname)
            The name of the output yaml config file where estimation results are written into.

        Returns
        -------
        lcm : MNLDiscreteChoiceModel which was used to fit
        """
        logger.debug('start: fit from configuration {}'.format(cfgname))
        lcm = cls.from_yaml(str_or_buffer=cfgname)
        lcm.fit(choosers, choosers[chosen_fname])
        lcm.report_fit()
        outcfgname = outcfgname or cfgname
        lcm.to_yaml(str_or_buffer=outcfgname)
        logger.debug('finish: fit into configuration {}'.format(outcfgname))
        return lcm

    @classmethod
    def predict_from_cfg(cls, choosers, cfgname=None, cfg=None,
                         debug=False):
        """
        Simulate choices for the specified choosers

        Parameters
        ----------
        choosers : DataFrame
            A dataframe of agents doing the choosing.
        alternatives : DataFrame
            A dataframe of locations which the choosers are locating in and
            which have a supply.
        cfgname : string
            The name of the yaml config file from which to read the discrete
            choice model.
        cfg: string
            an ordered yaml string of the model discrete choice model configuration.
            Used to read config from memory in lieu of loading cfgname from disk.
        alternative_ratio : float, optional
            Above the ratio of alternatives to choosers (default of 2.0),
            the alternatives will be sampled to meet this ratio
            (for performance reasons).
        debug : boolean, optional (default False)
            Whether to generate debug information on the model.

        Returns
        -------
        choices : pandas.Series
            Mapping of chooser ID to alternative ID. Some choosers
            will map to a nan value when there are not enough alternatives
            for all the choosers.
        lcm : MNLDiscreteChoiceModel which was used to predict
        """
        logger.debug('start: predict from configuration {}'.format(cfgname))
        if cfgname:
            lcm = cls.from_yaml(str_or_buffer=cfgname)
        elif cfg:
            lcm = cls.from_yaml(yaml_str=cfg)
        else:
            msg = 'predict_from_cfg requires a configuration via the cfgname or cfg arguments'
            logger.error(msg)
            raise ValueError(msg)

        #if len(alternatives) > len(choosers) * alternative_ratio:
        #    logger.info(
        #        ("Alternative ratio exceeded: %d alternatives "
        #         "and only %d choosers") %
        #        (len(alternatives), len(choosers)))
        #    idxes = np.random.choice(
        #        alternatives.index, size=int(len(choosers) *
        #                                     alternative_ratio),
        #        replace=False)
        #    alternatives = alternatives.loc[idxes]
        #    logger.info(
        #        "  after sampling %d alternatives are available\n" %
        #        len(alternatives))

        choice, probs = lcm.predict(choosers, debug=debug)
        print("Assigned %d choosers" % len(choice.dropna()))
        logger.debug('finish: predict from configuration {}'.format(cfgname))
        return choice, probs, lcm


#class MNLDiscreteChoiceModelGroup(DiscreteChoiceModel):
#    """
#    Manages a group of discrete choice models that refer to different
#    segments of choosers.

#    Model names must match the segment names after doing a pandas groupby.

#    Parameters
#    ----------
#    segmentation_col : str
#        Name of a column in the table of choosers. Will be used to perform
#        a pandas groupby on the choosers table.
#    remove_alts : bool, optional
#        Specify how to handle alternatives between prediction for different
#        models. If False, the alternatives table is not modified between
#        predictions. If True, alternatives that have been chosen
#        are removed from the alternatives table before doing another
#        round of prediction.
#    name : str, optional
#        A name that may be used in places to identify this group.

#    """
#    def __init__(self, segmentation_col, remove_alts=False, name=None):
#        self.segmentation_col = segmentation_col
#        self.remove_alts = remove_alts
#        self.name = name if name is not None else 'MNLDiscreteChoiceModelGroup'
#        self.models = {}

#    def add_model(self, model):
#        """
#        Add an MNLDiscreteChoiceModel instance.

#        Parameters
#        ----------
#        model : MNLDiscreteChoiceModel
#            Should have a ``.name`` attribute matching one of the segments
#            in the choosers table.

#        """
#        logger.debug(
#            'adding model {} to LCM group {}'.format(model.name, self.name))
#        self.models[model.name] = model

#    def add_model_from_params(
#            self, name, model_expression, sample_size,
#            probability_mode='full_product', choice_mode='individual',
#            choosers_fit_filters=None, choosers_predict_filters=None,
#            alts_fit_filters=None, alts_predict_filters=None,
#            interaction_predict_filters=None, estimation_sample_size=None,
#            prediction_sample_size=None, choice_column=None):
#        """
#        Add a model by passing parameters through to MNLDiscreteChoiceModel.

#        Parameters
#        ----------
#        name
#            Must match a segment in the choosers table.
#        model_expression : str, iterable, or dict
#            A patsy model expression. Should contain only a right-hand side.
#        sample_size : int
#            Number of choices to sample for estimating the model.
#        probability_mode : str, optional
#            Specify the method to use for calculating probabilities
#            during prediction.
#            Available string options are 'single_chooser' and 'full_product'.
#            In "single chooser" mode one agent is chosen for calculating
#            probabilities across all alternatives. In "full product" mode
#            probabilities are calculated for every chooser across all
#            alternatives.
#        choice_mode : str or callable, optional
#            Specify the method to use for making choices among alternatives.
#            Available string options are 'individual' and 'aggregate'.
#            In "individual" mode choices will be made separately for each
#            chooser. In "aggregate" mode choices are made for all choosers at
#            once. Aggregate mode implies that an alternative chosen by one
#            agent is unavailable to other agents and that the same
#            probabilities can be used for all choosers.
#        choosers_fit_filters : list of str, optional
#            Filters applied to choosers table before fitting the model.
#        choosers_predict_filters : list of str, optional
#            Filters applied to the choosers table before calculating
#            new data points.
#        alts_fit_filters : list of str, optional
#            Filters applied to the alternatives table before fitting the model.
#        alts_predict_filters : list of str, optional
#            Filters applied to the alternatives table before calculating
#            new data points.
#        interaction_predict_filters : list of str, optional
#            Filters applied to the merged choosers/alternatives table
#            before predicting agent choices.
#        estimation_sample_size : int, optional
#            Whether to sample choosers during estimation
#            (needs to be applied after choosers_fit_filters)
#        prediction_sample_size : int, optional
#            Whether (and how much) to sample alternatives during prediction.
#            Note that this can lead to multiple choosers picking the same
#            alternative.
#        choice_column : optional
#            Name of the column in the `alternatives` table that choosers
#            should choose. e.g. the 'building_id' column. If not provided
#            the alternatives index is used.

#        """
#        logger.debug('adding model {} to LCM group {}'.format(name, self.name))
#        self.models[name] = MNLDiscreteChoiceModel(
#            model_expression, sample_size,
#            probability_mode, choice_mode,
#            choosers_fit_filters, choosers_predict_filters,
#            alts_fit_filters, alts_predict_filters,
#            interaction_predict_filters, estimation_sample_size,
#            prediction_sample_size, choice_column, name)

#    def _iter_groups(self, data):
#        """
#        Iterate over the groups in `data` after grouping by
#        `segmentation_col`. Skips any groups for which there
#        is no model stored.

#        Yields tuples of (name, df) where name is the group key
#        and df is the group DataFrame.

#        Parameters
#        ----------
#        data : pandas.DataFrame
#            Must have a column with the same name as `segmentation_col`.

#        """
#        groups = data.groupby(self.segmentation_col)

#        for name, group in groups:
#            if name not in self.models:
#                continue
#            logger.debug(
#                'returning group {} in LCM group {}'.format(name, self.name))
#            yield name, group

#    def apply_fit_filters(self, choosers):
#        """
#        Filter `choosers` and `alternatives` for fitting.
#        This is done by filtering each submodel and concatenating
#        the results.

#        Parameters
#        ----------
#        choosers : pandas.DataFrame
#            Table describing the agents making choices, e.g. households.
#        alternatives : pandas.DataFrame
#            Table describing the things from which agents are choosing,
#            e.g. buildings.

#        Returns
#        -------
#        filtered_choosers, filtered_alts : pandas.DataFrame

#        """
#        ch = []

#        for name, df in self._iter_groups(choosers):
#            filtered_choosers, filtered_alts = \
#                self.models[name].apply_fit_filters(df, alternatives)
#            ch.append(filtered_choosers)
#            alts.append(filtered_alts)

#        return pd.concat(ch), pd.concat(alts)

#    def apply_predict_filters(self, choosers, alternatives):
#        """
#        Filter `choosers` and `alternatives` for prediction.
#        This is done by filtering each submodel and concatenating
#        the results.

#        Parameters
#        ----------
#        choosers : pandas.DataFrame
#            Table describing the agents making choices, e.g. households.
#        alternatives : pandas.DataFrame
#            Table describing the things from which agents are choosing,
#            e.g. buildings.

#        Returns
#        -------
#        filtered_choosers, filtered_alts : pandas.DataFrame

#        """
#        ch = []
#        alts = []

#        for name, df in self._iter_groups(choosers):
#            filtered_choosers, filtered_alts = \
#                self.models[name].apply_predict_filters(df, alternatives)
#            ch.append(filtered_choosers)
#            alts.append(filtered_alts)

#        filtered_choosers = pd.concat(ch)
#        filtered_alts = pd.concat(alts)

#        return filtered_choosers, filtered_alts.drop_duplicates()

#    def fit(self, choosers, alternatives, current_choice):
#        """
#        Fit and save models based on given data after segmenting
#        the `choosers` table.

#        Parameters
#        ----------
#        choosers : pandas.DataFrame
#            Table describing the agents making choices, e.g. households.
#            Must have a column with the same name as the .segmentation_col
#            attribute.
#        alternatives : pandas.DataFrame
#            Table describing the things from which agents are choosing,
#            e.g. buildings.
#        current_choice
#            Name of column in `choosers` that indicates which alternative
#            they have currently chosen.

#        Returns
#        -------
#        log_likelihoods : dict of dict
#            Keys will be model names and values will be dictionaries of
#            log-liklihood values as returned by MNLDiscreteChoiceModel.fit.

#        """
#        with log_start_finish(
#                'fit models in LCM group {}'.format(self.name), logger):
#            return {
#                name: self.models[name].fit(df, alternatives, current_choice)
#                for name, df in self._iter_groups(choosers)}

#    @property
#    def fitted(self):
#        """
#        Whether all models in the group have been fitted.

#        """
#        return (all(m.fitted for m in self.models.values())
#                if self.models else False)

#    def probabilities(self, choosers, alternatives):
#        """
#        Returns alternative probabilties for each chooser segment as
#        a dictionary keyed by segment name.

#        Parameters
#        ----------
#        choosers : pandas.DataFrame
#            Table describing the agents making choices, e.g. households.
#            Must have a column matching the .segmentation_col attribute.
#        alternatives : pandas.DataFrame
#            Table describing the things from which agents are choosing.

#        Returns
#        -------
#        probabilties : dict of pandas.Series

#        """
#        logger.debug(
#            'start: calculate probabilities in LCM group {}'.format(self.name))
#        probs = {}

#        for name, df in self._iter_groups(choosers):
#            probs[name] = self.models[name].probabilities(df, alternatives)

#        logger.debug(
#            'finish: calculate probabilities in LCM group {}'.format(
#                self.name))
#        return probs

#    def summed_probabilities(self, choosers, alternatives):
#        """
#        Returns the sum of probabilities for alternatives across all
#        chooser segments.

#        Parameters
#        ----------
#        choosers : pandas.DataFrame
#            Table describing the agents making choices, e.g. households.
#            Must have a column matching the .segmentation_col attribute.
#        alternatives : pandas.DataFrame
#            Table describing the things from which agents are choosing.

#        Returns
#        -------
#        probs : pandas.Series
#            Summed probabilities from each segment added together.

#        """
#        if len(alternatives) == 0 or len(choosers) == 0:
#            return pd.Series()

#        logger.debug(
#            'start: calculate summed probabilities in LCM group {}'.format(
#                self.name))
#        probs = []

#        for name, df in self._iter_groups(choosers):
#            probs.append(
#                self.models[name].summed_probabilities(df, alternatives))

#        add = tz.curry(pd.Series.add, fill_value=0)
#        probs = tz.reduce(add, probs)

#        logger.debug(
#            'finish: calculate summed probabilities in LCM group {}'.format(
#                self.name))
#        return probs

#    def predict(self, choosers, alternatives, debug=False):
#        """
#        Choose from among alternatives for a group of agents after
#        segmenting the `choosers` table.

#        Parameters
#        ----------
#        choosers : pandas.DataFrame
#            Table describing the agents making choices, e.g. households.
#            Must have a column matching the .segmentation_col attribute.
#        alternatives : pandas.DataFrame
#            Table describing the things from which agents are choosing.
#        debug : bool
#            If debug is set to true, will set the variable "sim_pdf" on
#            the object to store the probabilities for mapping of the
#            outcome.

#        Returns
#        -------
#        choices : pandas.Series
#            Mapping of chooser ID to alternative ID. Some choosers
#            will map to a nan value when there are not enough alternatives
#            for all the choosers.

#        """
#        logger.debug('start: predict models in LCM group {}'.format(self.name))
#        results = []

#        for name, df in self._iter_groups(choosers):
#            choices = self.models[name].predict(df, alternatives, debug=debug)
#            if self.remove_alts and len(alternatives) > 0:
#                alternatives = alternatives.loc[
#                    ~alternatives.index.isin(choices)]
#            results.append(choices)

#        logger.debug(
#            'finish: predict models in LCM group {}'.format(self.name))
#        return pd.concat(results) if results else pd.Series()

#    def choosers_columns_used(self):
#        """
#        Columns from the choosers table that are used for filtering.

#        """
#        return list(tz.unique(tz.concat(
#            m.choosers_columns_used() for m in self.models.values())))

#    def alts_columns_used(self):
#        """
#        Columns from the alternatives table that are used for filtering.

#        """
#        return list(tz.unique(tz.concat(
#            m.alts_columns_used() for m in self.models.values())))

#    def interaction_columns_used(self):
#        """
#        Columns from the interaction dataset used for filtering and in
#        the model. These may come originally from either the choosers or
#        alternatives tables.

#        """
#        return list(tz.unique(tz.concat(
#            m.interaction_columns_used() for m in self.models.values())))

#    def columns_used(self):
#        """
#        Columns from any table used in the model. May come from either
#        the choosers or alternatives tables.

#        """
#        return list(tz.unique(tz.concat(
#            m.columns_used() for m in self.models.values())))