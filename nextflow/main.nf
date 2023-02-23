
include { prepare_data } from './data_preparation.nf'
include { two_density } from './pipeline_double_density.nf'
include { one_density } from './pipeline_single_density.nf'


// IFL Parameters
feature_method = params.feature_method
single_method = params.single_method
double_method = params.double_method
config = file(params.configname)

process_py = file("python/src/process_diff.py")

workflow {

    main:
        prepare_data(params.path, params.extension)
        one_density(prepare_data.out[0], feature_method, single_method, config, process_py, params.path)
        two_density(prepare_data.out[1], feature_method, double_method, config, process_py, params.path)
        two_density[0].out.concat(one_density[0].out).collectFile(name: "${params.out}/benchmark.csv", skip: 1, keepHeader: true)
        two_density[1].out.concat(one_density[1].out).collectFile(name: "${params.out}/reconstruction.csv", skip: 1, keepHeader: true)
}
