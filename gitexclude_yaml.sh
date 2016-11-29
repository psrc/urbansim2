for FILE in configs/*.yaml; do
    source gitexclude.sh ${FILE}
done
