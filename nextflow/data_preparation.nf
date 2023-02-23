
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
        echo -e "X,Y,Z\n\$(cat ${base0}.txt)" > tmp${base0}.txt
        echo -e "X,Y,Z\n\$(cat ${base1}.txt)" > tmp${base1}.txt
        """
}


process append_columns_headers_rgb_label {

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

workflow prepare_txt_real_data {

    take:
        path

    main:
        paired = Channel.fromFilePairs(path + "{0,1}.txt")
        append_columns_headers(paired)
        append_columns_headers.out.set{pairedPointsclouds}
        pairedPointsclouds.map{it -> [[it[0], it[1]], [it[0], it[2]]]}.flatten().buffer(size: 2).set{pointClouds}

    emit:
        pairedPointsclouds
        pointClouds
}

workflow prepare_data {

    take:
        path
        extension

    main:
        if (extension == "ply"){
            paired = Channel.fromFilePairs(path + "*{0,1}.ply")
            from_ply_to_txt(paired).set{ply_files}
            ply_files.map{it -> [it[0].baseName.split("/")[-1], it[0], it[1]]}.set{pairedPointsclouds}
            ply_files.map{it -> [[it[0].baseName.split("/")[-1], it[0]], [it[0].baseName.split("/")[-1], it[1]]]}.flatten().buffer(size: 2).set{pointClouds}
        } else {
            paired = Channel.fromFilePairs(path + "{0,1}.txt")
            append_columns_headers_rgb_label(paired)
            append_columns_headers_rgb_label.out.set{pairedPointsclouds}
            pairedPointsclouds.map{it -> [[it[0], it[1]], [it[0], it[2]]]}.flatten().buffer(size: 2).set{pointClouds}
        }

    emit:
        pairedPointsclouds
        pointClouds
}
