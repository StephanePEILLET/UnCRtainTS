import datetime as dt
from itertools import compress
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor

# Launch date of Sentinel-2A
REFERENCE_DATE: dt.date = dt.datetime(*map(int, '2015-06-23'.split("-")), tzinfo=None).date()
# Strategies for positional encoding
PE_STRATEGIES = ['day-of-year', 'day-within-sequence', 'absolute', 'enumeration']


def get_position_for_positional_encoding(dates: List[dt.date], strategy: str) -> Tensor:
    """
    Function taken from https://github.com/prs-eth/U-TILISE/blob/main/lib/datasets/dataset_tools.py 

    Extracts the position index for every observation in an image time series, expressed as the number of days since
    a given reference date. The position indices will be used for sinusoidal positional encoding in the temporal encoder
    of U-TILISE.

    Args:
        dates:     list of datetime.date dates, acquisition dates for every observation in the sequence.
        strategy:  str, specifies the reference date. Choose among:
                        'day-of-year':          The position of each observation is expressed as the number of days
                                                since the 1st of January of the respective year, where the
                                                1st of January equals position 0 (i.e, seasonal information is
                                                implicitly encoded in the position).
                        'day-within-sequence':  The position of each observation is expressed relative to the first
                                                observation in the sequence, i.e., the first observation in the sequence
                                                is encoded as position 0 (i.e, seasonal information is not encoded in
                                                the position).
                        'absolute':             The position of each observation is expressed as the number of days
                                                since the reference date `REFERENCE_DATE`.
                        'enumeration':          Simple enumeration of the observations, i.e., 0, 1, 2, 3, etc.

    Returns:
        position:  torch.Tensor, number of days since a given reference date for every observation in the sequence.
    """

    if strategy == 'enumeration':
        position = torch.arange(0, len(dates))
    elif strategy == 'day-of-year':
        position = Tensor([(date - dt.date(date.year, 1, 1)).days for date in dates])
    elif strategy == 'day-within-sequence':
        position = Tensor([(date - dates[0]).days for date in dates])
    elif strategy == 'absolute':
        position = Tensor([(date - REFERENCE_DATE).days for date in dates])
    else:
        raise NotImplementedError(f'Unknown positional encoding strategy {strategy}.\n')

    return position