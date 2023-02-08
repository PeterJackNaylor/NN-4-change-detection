

py_file = file("python/src/optuna_trial.py")

process one_density_estimation {
    publishDir "${params.out}/single/${NAME}/", pattern: "*.png"
    label "gpu"
    clusterOptions  {DATANAME.contains("LyonS") ? "-jc gs-container_g1 -ac d=nvcr-pytorch-2204 -v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH" : 
    "-jc gpu-container_g1 -ac d=nvcr-pytorch-2204 -v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH"}    
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


// pyselect = file("python/src/selectbest.py")
// process selection {
//     publishDir "${params.out}/single/selection", mode: 'symlink'
//     input:
//         path(CSV)

//     output:
//         path("selected.csv")
//         path(CSV)
//     script:
//         """
//         python $pyselect $CSV
//         """
// }

process = file("python/src/process_diff.py")


process post_processing {
    label "gpu"
    clusterOptions  {DATANAME.contains("LyonS") ? "-jc gs-container_g1 -ac d=nvcr-pytorch-2204 -v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH" : 
    "-jc gpu-container_g1 -ac d=nvcr-pytorch-2204 -v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH"}   
    publishDir "${params.out}/single/${NAME}/", mode: 'symlink'
    input:
        tuple val(NAME), path(NPZ), path(WEIGHT), path(FILE0), path(FILE1), val(METHOD)

    output:
        tuple val(NAME), val("$METHOD"), path("${METHOD}*_results.npz")
        path("*.png")
    script:
        """
        python $process ${METHOD} ${WEIGHT} ${FILE0} ${FILE1} ${NPZ}
        """
}


process aggregate {
    publishDir "${params.out}/${METHOD}", mode: 'symlink'
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
        post_processing(one_density_estimation.out[0])
        aggregate(post_processing.out[0].groupTuple(by: [0, 1]))
    emit:
        aggregate.out[0]
}

workflow {
    one_density(data, fourier, method)
}
