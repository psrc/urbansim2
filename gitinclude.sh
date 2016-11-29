FILE=$1
if [[ -z ${FILE} ]]; then echo "No file specified."; exit; fi
if test ! -f ${FILE}; then echo "File ${FILE} does not exists."; exit; fi
cmd="git update-index --no-assume-unchanged ${FILE}"
echo $cmd
$cmd
