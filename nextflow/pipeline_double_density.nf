
py_file = file("python/src/optuna_trial.py")
process two_density_estimation {

    publishDir "${params.out}/double/${NAME}/", pattern: "*.png", overwrite: true
    label "gpu"

    input:
        tuple val(DATANAME), path(FILE)
        each FEATURE
        each METHOD
        path CONFIG

    output:
        tuple val(NAME),  path("$FNAME" + ".npz"), path("$FNAME" + ".pth"), path(FILE), val(METHOD)
        path("$FNAME" + "*.png")

    script:
        METH = "double_${METHOD}"
        NAME = "${DATANAME}__FEATUREMETHOD=${FEATURE}__METHOD=${METH}"
        FNAME = "${DATANAME}${FILE.baseName[-1]}__FEATUREMETHOD=${FEATURE}__METHOD=${METH}"
        """
        python $py_file \
            --csv0 $FILE \
            ${FEATURE} \
            --method $METHOD \
            --name $FNAME\
            --yaml_file $CONFIG \
            --method $METHOD
        """
}

process = file("python/src/process_diff.py")

process post_processing {

    label "gpu"
    publishDir "${params.out}/double/${NAME}/", mode: 'symlink', overwrite: true

    input:
        tuple val(NAME), path(NPZ), path(WEIGHTS), path(FILE), val(METHOD)
        path CONFIG

    output:
        path("*.csv")
        path("*.png")

    script:
        """
        python $process double_${METHOD[0]} ${WEIGHTS[0]} ${WEIGHTS[1]} ${FILE[0]} ${FILE[1]} ${NPZ[0]} ${NPZ[1]} ${CONFIG}
        """
}

workflow two_density {

    take:
        data
        feature_method
        method
        config

    main:
        two_density_estimation(data, feature_method, method, config)
        two_density_estimation.out[0].groupTuple().set{grouped}
        post_processing(grouped, config)

    emit:
        post_processing.out[0]
}
