
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

process one_density_estimation_ablation {
    publishDir "${params.out}/single/${NAME}/", pattern: "*.png", overwrite: true
    label "gpu"

    input:
        tuple val(DATANAME), file(FILE0), file(FILE1)
        each FEATURE
        each METHOD
        each LAMBDA
        path CONFIG

    output:
        tuple val(NAME), path("$NAME" + ".npz"), path("$NAME" + ".pth"), path(FILE0), path(FILE1), val(METHOD)
        path("$NAME" + "*.png")

    script:
        NAME = "${DATANAME}__FEATUREMETHOD=${FEATURE}__METHOD=${METHOD}__LAMBDA=${LAMBDA}"
        """
        python $py_file \
            --csv0 $FILE0 \
            --csv1 $FILE1 \
            $FEATURE \
            --method $METHOD \
            --name $NAME \
            --yaml_file $CONFIG \
            --lambda_reg $LAMBDA \
            --ablation
        """
}





process post_processing {
    label "gpu"
    publishDir "${params.out}/single/${NAME}/", mode: 'symlink', overwrite: true

    input:
        path PY
        tuple val(NAME), path(NPZ), path(WEIGHT), path(FILE0), path(FILE1), val(METHOD)
        path CONFIG

    output:
        path("*.csv")
        path("*.png")

    script:
        """
        python $PY single_${METHOD} ${WEIGHT} ${FILE0} ${FILE1} ${NPZ} ${CONFIG}
        """
}

process = file("python/src/reconstruction_mse.py")

process post_processing_mse {

    label "gpu"
    publishDir "${params.out}/single/${NAME}/", mode: 'symlink', overwrite: true

    input:
        tuple val(NAME), path(NPZ), path(WEIGHT), path(FILE0), path(FILE1), val(METHOD)
        path GTPATH
        path CONFIG

    output:
        path("*.csv")

    script:
        """
        python $process single_${METHOD[0]} ${WEIGHT} ${FILE0} ${FILE1} ${NPZ} ${GTPATH} ${CONFIG}
        """
}

workflow one_density_rd {

    take:
        paired_data
        feature_method
        method
        config
        py

    main:
        one_density_estimation(paired_data, feature_method, method, config)
        post_processing(py, one_density_estimation.out[0], config)

    emit:
        post_processing.out[0]
}


workflow one_density_ab {

    take:
        paired_data
        feature_method
        method
        config
        py
        path
        lambda

    main:
        one_density_estimation_ablation(paired_data, feature_method, method, lambda, config)
        post_processing(py, one_density_estimation_ablation.out[0], config)
        post_processing_mse(one_density_estimation_ablation.out[0],  path, config)

    emit:
        post_processing.out[0]
        post_processing_mse.out[0]
}


workflow one_density {

    take:
        paired_data
        feature_method
        method
        config
        py
        path

    main:
        one_density_estimation(paired_data, feature_method, method, config)
        post_processing(py, one_density_estimation.out[0], config)
        post_processing_mse(one_density_estimation.out[0],  path, config)

    emit:
        post_processing.out[0]
        post_processing_mse.out[0]
}
