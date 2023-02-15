
include { prepare_txt_real_data } from './data_preparation.nf'
include { two_density } from './pipeline_double_density.nf'
include { one_density } from './pipeline_single_density.nf'


// IFL Parameters
fourier = params.fourier
method = params.single_method
config = file(params.configname)


workflow {

    main:
        prepare_txt_real_data(params.path)

        two_density(pointClouds, fourier, config)
        one_density(pairedPointsclouds, fourier, method, config)

        two_density.out.concat(one_density.out).collectFile(name: "${params.out}/benchmark.csv", skip: 1, keepHeader: true)
}
