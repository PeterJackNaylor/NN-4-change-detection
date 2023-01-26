
include { two_density } from './pipeline_double_density.nf'
include { one_density } from './pipeline_single_density.nf'

// NeRF Parameters
scale = params.scale
fourier = params.fourier
norm = params.norm
lr = params.lr
mapping_size = params.mapping_size
act = params.act
epoch = params.epoch
arch = params.arch
wd = params.wd
lambda_t = params.lambda_t

// Chunk Parameters
ext = params.extension

// Data
paired_ply = Channel.fromFilePairs("data/full_data/*{0,1}.ply")
// paired_ply = Channel.fromFilePairs("data/LyonS/*{0,1}.ply")
paired_txt = Channel.fromFilePairs("data/clippeddata/clippedMarco{0,1}.txt")

process from_ply_to_txt {
    input:
        tuple val(key), file(paired_file)
    output:
        path("*{0,1}.txt")
    script:
        pyfile = file("python/src/opening_ply.py")
        base0 = paired_file[0].baseName
        base1 = paired_file[1].baseName
        """
        python $pyfile ${base0}.ply ${base1}.ply
        """
}

process append_columns_headers {
    input:
        tuple val(key), file(paired_file)
    output:
        tuple val(key), file("tmp*0.txt"), file("tmp*1.txt")

    script:
        base0 = paired_file[0].baseName
        base1 = paired_file[1].baseName
        """
        echo -e "X,Y,Z,R,G,B,label_ch\n\$(cat ${base0}.txt)" > tmp${base0}.txt
        echo -e "X,Y,Z,R,G,B,label_ch\n\$(cat ${base1}.txt)" > tmp${base1}.txt
        """
}


process final_table {
    publishDir "${params.out}"
    input:
        file(results)
    output:
        file("benchmark.csv")
    script:
        py_file = file("python/src/regroup_csv.py")
        """
        python $py_file

        """
}


workflow {
    main:
        if (ext == "ply"){
            from_ply_to_txt(paired_ply).set{pointClouds}
            pointClouds.buffer(size: 2).map{it -> [it[0][0], it[0][1], it[1][1]]}.set{pairedPointsclouds}
        } else {
            append_columns_headers(paired_txt)

            append_columns_headers.out.set{pairedPointsclouds}
            pairedPointsclouds.map{it -> [[it[0], it[1]], [it[0], it[2]]]}.flatten().buffer(size: 2).set{pointClouds}
        }
        two_density(pointClouds, scale, fourier, mapping_size, norm, arch, lr, wd, act, epoch)
        one_density(pairedPointsclouds, scale, fourier, mapping_size, norm, arch, lr, wd, lambda_t, act, epoch)
        two_density.out.concat(one_density.out).collect().set{results}
        final_table(results)
}
