
include { two_density } from './pipeline_double_density.nf'
include { one_density } from './pipeline_single_density.nf'

// NeRF Parameters
fourier = params.fourier
method = params.single_method
config = file(params.configname)
// Chunk Parameters
ext = params.extension

// Data
paired_ply = Channel.fromFilePairs(params.path + "*{0,1}.ply")
paired_txt = Channel.fromFilePairs(params.path + "{0,1}.txt")

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
            from_ply_to_txt(paired_ply).set{ply_files}
            ply_files.map{it -> [it[0].baseName.split("/")[-1], it[0], it[1]]}.set{pairedPointsclouds}
            ply_files.map{it -> [[it[0].baseName.split("/")[-1], it[0]], [it[0].baseName.split("/")[-1], it[1]]]}.flatten().buffer(size: 2).set{pointClouds}
        } else {
            append_columns_headers(paired_txt)

            append_columns_headers.out.set{pairedPointsclouds}
            pairedPointsclouds.map{it -> [[it[0], it[1]], [it[0], it[2]]]}.flatten().buffer(size: 2).set{pointClouds}
        }

        two_density(pointClouds, fourier, config)
        one_density(pairedPointsclouds, fourier, method, config)
        two_density.out.concat(one_density.out).collect().set{results}
        final_table(results)
}
