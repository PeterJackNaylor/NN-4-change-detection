
full_data_home:
	nextflow run nextflow/main.nf -resume --extension ply

full_data:
	nextflow run nextflow/main.nf -params-file exp_config/paper.yaml -resume -profile raiden

full_data_test:
	nextflow run nextflow/main.nf -params-file exp_config/test_raiden.yaml -resume -profile raiden

clipped:
	nextflow run nextflow/main.nf -resume --extension txt -profile raiden

clipped_home:
	nextflow run nextflow/main.nf -params-file exp_config/home.yaml -resume
