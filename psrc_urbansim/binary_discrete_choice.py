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
import pylogit
import statsmodels.api as sm
logger = logging.getLogger(__name__)

class DiscreteChoiceModel(object):
    """
    Abstract base class for discrete choice models.
    """
    __metaclass__ = abc.ABCMeta

    #@staticmethod
    #def _check_prob_choice_mode_compat(probability_mode, choice_mode):
    #    """
    #    Check that the probability and choice modes are compatibly with
    #    each other. Currently 'single_chooser' must be paired with
    #    'aggregate' and 'full_product' must be paired with 'individual'.
    #    """
    #    if (probability_mode == 'full_product' and
    #            choice_mode == 'aggregate'):
    #        raise ValueError(
    #            "'full_product' probability mode is not compatible with "
    #            "'aggregate' choice mode")

    #    if (probability_mode == 'single_chooser' and
    #            choice_mode == 'individual'):
    #        raise ValueError(
    #            "'single_chooser' probability mode is not compatible with "
    #            "'individual' choice mode")

    #@staticmethod
    #def _check_prob_mode_interaction_compat(
    #        probability_mode, interaction_predict_filters):
    #    """
    #    The 'full_product' probability mode is currently incompatible with
    #    post-interaction prediction filters, so make sure we don't have
    #    both of those.
    #    """
    #    if (interaction_predict_filters is not None and
    #            probability_mode == 'full_product'):
    #        raise ValueError(
    #            "interaction filters may not be used in "
    #            "'full_product' mode")

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

    #@abc.abstractmethod
    #def alts_columns_used(self):
    #    pass

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
    Based on binaryl logit.
    Parameters
    ----------
    model_expression : str, iterable, or dict
        A patsy model expression. Should contain only a right-hand side.
    choosers_fit_filters : list of str, optional
        Filters applied to choosers table before fitting the model.
    choosers_predict_filters : list of str, optional
        Filters applied to the choosers table before calculating
        new data points.
    interaction_predict_filters : list of str, optional
        Filters applied to the merged choosers/alternatives table
        before predicting agent choices.
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
            interaction_predict_filters=None,
            choice_column=None, name=None):
        self.model_expression = model_expression
        self.choosers_fit_filters = choosers_fit_filters
        self.choosers_predict_filters = choosers_predict_filters
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
            interaction_predict_filters=cfg.get(
                'interaction_predict_filters', None),
            choice_column=cfg.get('choice_column', None),
            name=cfg.get('name', None)
        )

        if cfg.get('log_likelihoods', None):
            model.log_likelihoods = cfg['log_likelihoods']
        if cfg.get('fit_parameters', None):
            model.fit_parameters = pd.DataFrame(cfg['fit_parameters'])

        logger.debug('loaded binary logit model {} from YAML'.format(model.name))
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
        Returns
        -------
        filtered_choosers : pandas.DataFrame
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
        Returns
        -------
        filtered_choosers : pandas.DataFrame
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

        if len(choosers) != model_design.to_numpy().shape[0]:
            raise ModelEvaluationError(
                'Estimated data does not have the same length as input.  '
                'This suggests there are null values in one or more of '
                'the input columns.')
        logit = sm.Logit(current_choice, model_design)
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

        if len(choosers) != model_design.to_numpy().shape[0]:
            raise ModelEvaluationError(
                'Simulated data does not have the same length as input.  '
                'This suggests there are null values in one or more of '
                'the input columns.')

        # get the order of the coefficients in the same order as the
        # columns in the design matrix
        coeffs = [self.fit_parameters['Coefficient'][x]
                  for x in model_design.columns]

        # Constructor requires and observation column, but since we are not estimating any will do so using constant. 
        logit = sm.Logit(model_design.loc[:,0], model_design)

        # Get the prediction probabilities for each chooser
        return pd.DataFrame(logit.predict(coeffs), columns=['probability'], index=model_design.index)
       
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
            return (pd.Series(), pd.Series())

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

    #def alts_columns_used(self):
    #    """
    #    Columns from the alternatives table that are used for filtering.
    #    """
    #    return list(tz.unique(tz.concatv(
    #        util.columns_in_filters(self.alts_predict_filters),
    #        util.columns_in_filters(self.alts_fit_filters))))

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
            #self.alts_columns_used(),
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
        cfgname : string
            The name of the yaml config file from which to read the discrete
            choice model.
        cfg: string
            an ordered yaml string of the model discrete choice model configuration.
            Used to read config from memory in lieu of loading cfgname from disk.
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


