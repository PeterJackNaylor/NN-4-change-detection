
include { aggregate } from './pipeline_single_density.nf'

py_file = file("python/src/main.py")
process two_density_estimation {
    label 'gpu'
    input:
        tuple val(DATANAME), path(FILE)
        each SCALE
        each FOUR
        each MAPPINGSIZE
        each NORM
        each ARCH
        each LR
        each WD
        each ACT
        each EPOCH

    output:
        path("$NAME" + ".csv")
        tuple val(NAME), val(DATANAME), path("$NAME" + ".npz"), path("$NAME" + ".pth"), path(FILE)
        path("$NAME" + ".png")

    script:
        NAME = "${FILE.baseName}__SCALE=${SCALE}__FOUR=${FOUR}__NORM=${NORM}__ARCH=${ARCH}__LR=${LR}__WD=${WD}__ACT=${ACT}__MAPPINGSIZE=${MAPPINGSIZE}_double"
        """
        python $py_file \
            --csv0 $FILE \
            --epochs $EPOCH \
            --scale $SCALE \
            --mapping_size $MAPPINGSIZE \
            $FOUR \
            --normalize $NORM \
            --arch $ARCH\
            --lr $LR \
            --wd $WD \
            --activation $ACT\
            --name $NAME\
            --workers 8
        """
}

pyselect = file("python/src/selectbest.py")
process selection {
    publishDir "${params.out}/double/selection", mode: 'symlink'
    input:
        path(CSV)

    output:
        path("selected.csv")
        path(CSV)
    script:
        """
        python $pyselect $CSV
        """
}

process = file("python/src/process_diff.py")


process post_processing {
    label 'gpu'
    publishDir "${params.out}/double/${DATANAMES}/", mode: 'symlink'
    input:
        tuple val(NAMES), val(DATANAMES), path(NPZ), path(WEIGHTS), path(FILE), val(CHUNK_ID), val(METHOD)

    output:
        tuple val(DATANAMES), val("${METHOD[0]}"), path("*${DATANAMES[0]}*_results.npz")
        path("*.png")

    script:
        """
        python $process ${METHOD[0]} ${WEIGHTS[0]} ${WEIGHTS[1]} ${FILE[0]} ${FILE[1]} ${NPZ[0]} ${NPZ[1]}
        """
}


workflow two_density {
    take:
        data
        scale
        fourier
        mapping_size
        norm
        arch
        lr
        wd
        act
        epoch
    main:
        two_density_estimation(data, scale, fourier, mapping_size, norm, arch, lr, wd, act, epoch)
        two_density_estimation.out[0].collectFile(name:"together.csv", keepHeader: true, skip:1).set{training_scores}
        selection(training_scores)
        selection.out[0] .splitCsv(skip:1, sep: ',')
            .set{selected}
        two_density_estimation.out[1].join(selected, by: 0).set{test}
        test.groupTuple(by: [1]).set{fused}
        post_processing(fused)
        aggregate(post_processing.out[0].groupTuple(by: [0, 1]))
    emit:
        aggregate.out[0]
}


data = Channel.fromPath("LyonN4/*.txt")
scale = [0.5] //0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
fourier = ["--fourier"]
norm = ["one_minus"]
lr = [0.1] //0.0001, 0.001, 0.01, 0.1, 1.0]
act = ["relu"]


workflow {
    main:
        two_density(data, scale, fourier, norm, lr, act)
}
