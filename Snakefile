# run the pipeline
rule pipeline:
    conda:
        "environment.yml"
    output:
        "src/data/pipeline/sample.pkl"
    cache:
        True
    script:
        "src/scripts/bioverse_pipeline.py"