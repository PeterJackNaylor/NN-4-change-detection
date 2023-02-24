
include { prepare_data } from './data_preparation.nf'
include { one_density_ab } from './pipeline_single_density.nf'


// IFL Parameters
feature_method = params.feature_method
single_method = params.single_method
lambda = params.lambda
config = file(params.configname)
datapath = file(params.gtpath)
process_py = file("python/src/process_diff.py")

workflow {

    main:
        prepare_data(params.path, params.extension)
        one_density_ab(prepare_data.out[0], feature_method, single_method, config, process_py, datapath, lambda)
        one_density_ab.out[0].collectFile(name: "${params.out}/benchmark_ablation.csv", skip: 1, keepHeader: true)
        one_density_ab.out[1].collectFile(name: "${params.out}/reconstruction_ablation.csv", skip: 1, keepHeader: true)
}
