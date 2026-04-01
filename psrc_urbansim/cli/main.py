import sys
from psrc_urbansim.cli import CLI
from psrc_urbansim.cli import run_simulation
from psrc_urbansim.cli import run_allocation
from psrc_urbansim.cli import run_indicators
#from network_builder.cli import build_transit_segments_parallel

from psrc_urbansim import __version__, __doc__

def main():
    run_model = CLI(version=__version__, description=__doc__)
    run_model.add_subcommand(
        name="run_simulation",
        args_func=run_simulation.add_run_args,
        exec_func=run_simulation.run,
        description=run_simulation.run.__doc__,
    )

    run_model.add_subcommand(
        name="run_allocation",
        args_func=run_allocation.add_run_args,
        exec_func=run_allocation.run,
        description=run_allocation.run.__doc__,
    )

    run_model.add_subcommand(
        name="run_indicators",
        args_func=run_indicators.add_run_args,
        exec_func=run_indicators.run,
        description=run_indicators.run.__doc__,
    )

    sys.exit(run_model.execute())