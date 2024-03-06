# run the pipeline
rule pipeline:
    conda:
        "environment.yml"
    output:
        "src/data/pipeline/sample.dll"
        "src/data/pipeline/data.dll"
        "src/data/pipeline/grid_flife_nuv.dll"
    cache:
        True
    script:
        "src/scripts/bioverse_pipeline.py"