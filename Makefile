
full_data_home:
	nextflow run nextflow/main.nf -resume --extension ply

full_data:
	nextflow run nextflow/main.nf -resume --extension ply -profile raiden

clipped:
	nextflow run nextflow/main.nf -resume --extension txt -profile raiden

clipped_home:
	nextflow run nextflow/main.nf -resume --extension txt