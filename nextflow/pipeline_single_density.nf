

py_file = file("python/src/main.py")

process one_density_estimation {
    label 'gpu'
    input:
        tuple val(DATANAME), file(FILE0), file(FILE1)
        each SCALE
        each FOUR
        each MAPPINGSIZE
        each NORM
        each ARCH
        each LR
        each WD
        each LAMBDA_T
        each ACT
        each EPOCH
        each act_last


    output:
        path("$NAME" + ".csv")
        tuple val(NAME), val(DATANAME), path("$NAME" + ".npz"), path("$NAME" + ".pth"), path(FILE0), path(FILE1)
        tuple path("$NAME" + "0.png"), path("$NAME" + "1.png")

    script:
        CHUNK_ID = FILE0.baseName.split("-")[0]
        if (LAMBDA_T == 0.0){
            postfix = "single"
        }else{
            postfix = "singleRegulated"
        }
        NAME = "${CHUNK_ID}-${DATANAME}__SCALE=${SCALE}__FOUR=${FOUR}__NORM=${NORM}__ARCH=${ARCH}__LR=${LR}__WD=${WD}__ACT=${ACT}__MAPPINGSIZE=${MAPPINGSIZE}__REGUL=${LAMBDA_T}_${postfix}"
        """
        python $py_file \
            --csv0 $FILE0 \
            --csv1 $FILE1 \
            --epochs $EPOCH \
            --scale $SCALE \
            --mapping_size $MAPPINGSIZE \
            $FOUR \
            --normalize $NORM \
            --arch $ARCH\
            --lr $LR \
            --wd $WD \
            --lambda_t ${LAMBDA_T} \
            --activation $ACT \
            --name $NAME \
            --workers 8 \
            --act_last ${act_last}
        """
}


pyselect = file("python/src/selectbest.py")
process selection {
    publishDir "${params.out}/single/selection", mode: 'symlink'
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
    publishDir "${params.out}/single/${DATANAMES}/", mode: 'symlink'
    input:
        tuple val(NAMES), val(DATANAMES), path(NPZ), path(WEIGHTS), path(FILE0), path(FILE1), val(CHUNK_ID), val(METHOD)

    output:
        tuple val(DATANAMES), val(METHOD), path("*${DATANAMES}*_results.npz")
        path("*.png")
    script:
        """
        python $process ${METHOD} ${WEIGHTS} ${FILE0} ${FILE1} ${NPZ}
        """
}


process aggregate {
    publishDir "${params.out}/${METHOD}/", mode: 'symlink'
    input:
        tuple val(DATANAME), val(METHOD), path(NPZ)
    output:
        path("${DATANAME}_${METHOD}.csv")
        path("${DATANAME}_${METHOD}_chunkinfo.csv")

    script:
        py_file = file("python/src/aggregate.py")
        """
        python $py_file $DATANAME $METHOD
        """
}

data = Channel.fromFilePairs("LyonN4/*{0,1}.txt")
scale = [0.5] //0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
fourier = ["--fourier"]
norm = ["one_minus"]
lr = [0.1] //0.0001, 0.001, 0.01, 0.1, 1.0]
act = ["relu"]


workflow one_density {
    take:
        paired_data
        scale
        fourier
        mapping_size
        norm
        arch
        lr
        wd
        lambda_t
        act
        epoch
        act_last
    main:
        one_density_estimation(paired_data, scale, fourier, mapping_size, norm, arch, lr, wd, lambda_t, act, epoch, act_last)
        one_density_estimation.out[0].collectFile(name:"together.csv", keepHeader: true, skip:1).set{training_scores}
        selection(training_scores)
        selection.out[0] .splitCsv(skip:1, sep: ',')
            .set{selected}
        one_density_estimation.out[1].join(selected, by: 0).set{fused}
        post_processing(fused)
        aggregate(post_processing.out[0].groupTuple(by: [0, 1]))
    emit:
        aggregate.out[0]
}

workflow {
    one_density(data, scale, fourier, norm, lr, act)
}
