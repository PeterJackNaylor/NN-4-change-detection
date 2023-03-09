

paper_cluster:
	nextflow run nextflow/main.nf -params-file exp_config/paper.yaml -resume -profile cluster

ablation_cluster:
	nextflow run nextflow/ablation_study_hp.nf -params-file exp_config/paper_ablation.yaml -resume -profile cluster

paper_home:
	nextflow run nextflow/main.nf -params-file exp_config/home.yaml -resume -profile local

ablation_home:
	nextflow run nextflow/ablation_study_hp.nf -params-file exp_config/home_ablation.yaml -resume -profile local

looting_home:
	nextflow run nextflow/main_real_data.nf -params-file exp_config/home_looting.yaml -resume -profile local
