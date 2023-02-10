
py_file = file("python/src/optuna_trial.py")

process one_density_estimation {
    publishDir "${params.out}/single/${NAME}/", pattern: "*.png"
    label "gpu"

    input:
        tuple val(DATANAME), file(FILE0), file(FILE1)
        each FOUR
        each METHOD
        path CONFIG

    output:
        // path("$NAME" + ".csv")
        tuple val(NAME), path("$NAME" + ".npz"), path("$NAME" + ".pth"), path(FILE0), path(FILE1), val(METHOD)
        path("$NAME" + "*.png")

    script:
        NAME = "${DATANAME}__FOUR=${FOUR}__METHOD=${METHOD}"
        """
        python $py_file \
            --csv0 $FILE0 \
            --csv1 $FILE1 \
            $FOUR \
            --name $NAME \
            --yaml_file $CONFIG

        """
}

process = file("python/src/process_diff.py")

process post_processing {
    label "gpu"
    publishDir "${params.out}/single/${NAME}/", mode: 'symlink'

    input:
        tuple val(NAME), path(NPZ), path(WEIGHT), path(FILE0), path(FILE1), val(METHOD)
        path CONFIG

    output:
        path("*.csv")
        path("*.png")

    script:
        """
        python $process ${METHOD} ${WEIGHT} ${FILE0} ${FILE1} ${NPZ} ${CONFIG}
        """
}

data = Channel.fromFilePairs("data/clippeddata/*{0,1}.txt")
fourier = ["--fourier"]
method = ["None", "L1_diff"]
config = Channel.fromPath("exp_config/home.yaml")

workflow one_density {
    take:
        paired_data
        fourier
        method
        config

    main:
        one_density_estimation(paired_data, fourier, method, config)
        post_processing(one_density_estimation.out[0], config)
    emit:
        post_processing.out[0]
}

workflow {
    one_density(data, fourier, method)
}
