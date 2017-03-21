# Bash script to assign a run id/number
# -------------------------------------
# Usage: 
# ./get_run_id.sh [prefix]
#
# If network drive is found, in other words if we're on the
# local network, use the global run numbering. Otherwise
# use numbering local to this machine and prepend the
# machine name so we know the difference.

# Network location of run number counters
RUN_NUMBER_DIR=/r/Urbansim2/run-numbers

# Should be no need to change anything below this line.
# -----------------------------------------------------

# Use a run prefix if user supplied one, e.g. luv -> luv-0001
if [ -n "$1" ]
then
    PREFIX="$1"
else
    PREFIX="run"
fi

# If network location not found, prepend hostname and store in ~/.run-numbers
if [ ! -d $RUN_NUMBER_DIR ]
then
    mkdir -p ~/.run-numbers
    RUN_NUMBER_DIR=~/.run-numbers
    PREFIX=$(hostname)-$PREFIX
fi

cd $RUN_NUMBER_DIR
run_number=$PREFIX-number.txt

# Attempt to get run number five times
for loop in 1 2 3 4 5
do
    # use a lockfile to prevent script from running from multiple threads
    # (if something goes wrong, just delete $RUN_NUMBER_DIR/lockfile)
    if ( set -o noclobber; echo "$$" > lockfile-delete-me.txt) 2> /dev/null; then
        trap 'rm -f "$lockfile"; exit $?' INT TERM EXIT

        # read last run number from file, if it exists
        if [ -e $run_number ]
        then
            OLD_RUN=`cat "$run_number"`
        fi
        
        # calculate and save new run number
        NEW_RUN=$((OLD_RUN + 1))
        echo $NEW_RUN > "$run_number"
        printf "%s-%04d" "$PREFIX" "$NEW_RUN"

        # clean up after yourself, and release your lock trap
        rm -f lockfile-delete-me.txt
        trap - INT TERM EXIT
        exit 0
    else
        # lock exists, wait and try again in a second
        (>&2 echo Script is locked, trying again...)
        sleep 2
    fi
done

(>&2 echo "Lockfile $RUN_NUMBER_DIR/$lockfile exists, aborting")
exit 2

