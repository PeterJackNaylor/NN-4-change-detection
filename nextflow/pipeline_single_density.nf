
py_file = file("python/src/optuna_trial.py")

process one_density_estimation {
    publishDir "${params.out}/single/${NAME}/", pattern: "*.png", overwrite: true
    label "gpu"

    input:
        tuple val(DATANAME), file(FILE0), file(FILE1)
        each FEATURE
        each METHOD
        path CONFIG

    output:
        tuple val(NAME), path("$NAME" + ".npz"), path("$NAME" + ".pth"), path(FILE0), path(FILE1), val(METHOD)
        path("$NAME" + "*.png")

    script:
        NAME = "${DATANAME}__FEATUREMETHOD=${FEATURE}__METHOD=${METHOD}"
        """
        python $py_file \
            --csv0 $FILE0 \
            --csv1 $FILE1 \
            $FEATURE \
            --method $METHOD \
            --name $NAME \
            --yaml_file $CONFIG
        """
}

process = file("python/src/process_diff.py")

process post_processing {
    label "gpu"
    publishDir "${params.out}/single/${NAME}/", mode: 'symlink', overwrite: true

    input:
        tuple val(NAME), path(NPZ), path(WEIGHT), path(FILE0), path(FILE1), val(METHOD)
        path CONFIG

    output:
        path("*.csv")
        path("*.png")

    script:
        """
        python $process single_${METHOD} ${WEIGHT} ${FILE0} ${FILE1} ${NPZ} ${CONFIG}
        """
}

workflow one_density {

    take:
        paired_data
        feature_method
        method
        config

    main:
        one_density_estimation(paired_data, feature_method, method, config)
        post_processing(one_density_estimation.out[0], config)

    emit:
        post_processing.out[0]
}
