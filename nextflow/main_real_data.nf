
include { prepare_txt_real_data } from './data_preparation.nf'
include { two_density_rd } from './pipeline_double_density.nf'
include { one_density_rd } from './pipeline_single_density.nf'


// IFL Parameters
feature_method = params.feature_method
single_method = params.single_method
double_method = params.double_method
config = file(params.configname)

process_py = file("python/src/process_diff_cambodia.py")

workflow {

    main:
        prepare_txt_real_data(params.path)
        prepare_txt_real_data.out[0] .set{ppC}
        prepare_txt_real_data.out[1] .set{pC}


        two_density_rd(pC, feature_method, double_method, config, process_py)
        one_density_rd(ppC, feature_method, single_method, config, process_py)

        two_density_rd.out.concat(one_density_rd.out).collectFile(name: "${params.out}/benchmark.csv", skip: 1, keepHeader: true)
}
