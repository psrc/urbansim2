from urbansim.models.transition import TransitionModel
import logging
from urbansim.utils.logutil import log_start_finish
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PSRCTabularTotalsTransition(TransitionModel):
    """
    There is a bug in the function _update_linked_table in urbansim/urbansim/models/transition.py
    (if a household is both, copied and removed, those copied households end up without any person).
    Thus, this class tries to go around it.
    """
    
    def transition(self, data, year, linked_tables=None):
        """
        Add or remove rows from a table based on population targets.

        Parameters
        ----------
        data : pandas.DataFrame
            Rows will be removed from or added to this table.
        year : int
            Year number that will be passed to `transitioner`.
        linked_tables : dict of tuple, optional
            Dictionary of (table, 'column name') pairs. The column name
            should match the index of `data`. Indexes in `data` that
            are copied or removed will also be copied and removed in
            linked tables. They dictionary keys are used in the
            returned `updated_links`.

        Returns
        -------
        updated : pandas.DataFrame
            Table with rows removed or added.
        added : pandas.Series
            Indexes of new rows in `updated`.
        updated_links : dict of pandas.DataFrame

        """
        logger.debug('start: transition')
        linked_tables = linked_tables or {}
        updated_links = {}

        with log_start_finish('add/remove rows', logger):
            updated, added, copied, removed = self.transitioner(data, year)

        for table_name, (table, col) in linked_tables.items():
            logger.debug('updating linked table {}'.format(table_name))
            updated_links[table_name] = \
                _update_linked_table(table, col, added, copied, removed)

        logger.debug('finish: transition')
        return updated, added, updated_links
    
def _update_linked_table(table, col_name, added, copied, removed):
    """
    This contains a bug fix found in urbansim/urbansim/models/transition.py
    
    Copy and update rows in a table that has a column referencing another
    table that has had rows added via copying.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to update with new or removed rows.
    col_name : str
        Name of column in `table` that corresponds to the index values
        in `copied` and `removed`.
    added : pandas.Index
        Indexes of rows that are new in the linked table.
    copied : pandas.Index
        Indexes of rows that were copied to make new rows in linked table.
    removed : pandas.Index
        Indexes of rows that were removed from the linked table.

    Returns
    -------
    updated : pandas.DataFrame

    """
    logger.debug('start: update linked table after transition')

    # handle removals
    # HS fix: the removal needs to happen independently of creating new rows,
    # because some of the removed rows might be contained in the "copied" object.
    # Thus, we use here "table_new" instead of overwriting the "table" object.
    # The original "table" object can be then used in the id_map.merge() command.
    table_new = table.loc[~table[col_name].isin(set(removed))] 
    if (added is None or len(added) == 0):
        return table_new

    # map new IDs to the IDs from which they were copied
    id_map = pd.concat([pd.Series(copied, name=col_name), pd.Series(added, name='temp_id')], axis=1)

    # join to linked table and assign new id
    new_rows = id_map.merge(table, on=col_name) # this must happen on the original table (not the one after removing rows) 
    new_rows.drop(col_name, axis=1, inplace=True)
    new_rows.rename(columns={'temp_id': col_name}, inplace=True)

    # index the new rows
    starting_index = table_new.index.values.max() + 1
    new_rows.index = np.arange(starting_index, starting_index + len(new_rows), dtype=int)

    logger.debug('finish: update linked table after transition')
    return pd.concat([table_new, new_rows])
