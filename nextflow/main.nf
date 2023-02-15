
include { prepare_data } from './data_preparation.nf'
include { two_density } from './pipeline_double_density.nf'
include { one_density } from './pipeline_single_density.nf'


// IFL Parameters
fourier = params.fourier
single_method = params.single_method
double_method = params.double_method
config = file(params.configname)


workflow {

    main:
        prepare_data(params.path, params.extension)
        one_density(prepare_data.out[0], fourier, single_method, config)
        two_density(prepare_data.out[1], fourier, double_method, config)
        two_density.out.concat(one_density.out).collectFile(name: "${params.out}/benchmark.csv", skip: 1, keepHeader: true)
}
