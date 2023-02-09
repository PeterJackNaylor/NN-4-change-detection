
include { aggregate } from './pipeline_single_density.nf'

py_file = file("python/src/optuna_trial.py")
process two_density_estimation {
    publishDir "${params.out}/double/${NAME}/", pattern: "*.png"
    label "gpu"
    input:
        tuple val(DATANAME), path(FILE)
        each FOUR
        path CONFIG

    output:
        tuple val(NAME),  path("$FNAME" + ".npz"), path("$FNAME" + ".pth"), path(FILE)
        path("$FNAME" + ".png")

    script:
        NAME = "${DATANAME}__FOUR=${FOUR}_double"
        FNAME = "${FILE.baseName}__FOUR=${FOUR}_double"
        """
        python $py_file \
            --csv0 $FILE \
            $FOUR \
            --name $FNAME\
            --yaml_file $CONFIG \
            --method double
        """
}

process = file("python/src/process_diff.py")


process post_processing {
    label "gpu"
    publishDir "${params.out}/double/${NAME}/", mode: 'symlink'
    input:
        tuple val(NAME), path(NPZ), path(WEIGHTS), path(FILE)
        path CONFIG

    output:
        tuple val(NAME), val("double"), path("double*_results.npz")
        path("*.png")

    script:
        """
        python $process double ${WEIGHTS[0]} ${WEIGHTS[1]} ${FILE[0]} ${FILE[1]} ${NPZ[0]} ${NPZ[1]} ${CONFIG}
        """
}


workflow two_density {
    take:
        data
        fourier
        config
    main:
        two_density_estimation(data, fourier, config)
        two_density_estimation.out[0].groupTuple().set{fused}
        post_processing(fused, config)
        aggregate(post_processing.out[0].groupTuple(by: [0, 1]))
    emit:
        aggregate.out[0]
}


data = Channel.fromPath("LyonN4/*.txt")
fourier = ["--fourier"]



workflow {
    main:
        two_density(data, fourier)
}
