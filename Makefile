

full_data:
	nextflow run nextflow/main.nf -params-file exp_config/paper.yaml -resume -profile raiden

clipped:
	nextflow run nextflow/main.nf -resume --extension txt -profile raiden

full_data_test:
	nextflow run nextflow/main.nf -params-file exp_config/test_raiden.yaml -resume -profile raiden

ablation_raiden:
	nextflow run nextflow/ablation_study_hp.nf -params-file exp_config/paper_ablation.yaml -resume -profile raiden


full_data_kuma:
	nextflow run nextflow/main.nf -params-file exp_config/kuma.yaml -resume -profile kuma

looting_kuma:
	nextflow run nextflow/main_real_data.nf -params-file exp_config/kuma_looting.yaml -resume -profile kuma


full_data_home:
	nextflow run nextflow/main.nf -resume --extension ply

clipped_home:
	nextflow run nextflow/main.nf -params-file exp_config/home.yaml -resume -profile local

ablation_home:
	nextflow run nextflow/ablation_study_hp.nf -params-file exp_config/home_ablation.yaml -resume -profile local

looting_home:
	nextflow run nextflow/main_real_data.nf -params-file exp_config/looting_home.yaml -resume -profile local
