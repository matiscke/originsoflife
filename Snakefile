# run the pipeline
rule pipeline:
    conda:
        "environment.yml"
    output:
        "src/data/pipeline/sample.dll",
        "src/data/pipeline/sample_M.dll",
        "src/data/pipeline/sample_FGK.dll",
        "src/data/pipeline/data.dll",
        "src/data/pipeline/data_M.dll",
        "src/data/pipeline/data_FGK.dll",
        "src/data/pipeline/grid_flife_nuv.dll",
        "src/data/pipeline/grid_flife_nuv_M.dll",
        "src/data/pipeline/grid_flife_nuv_FGK.dll",
        "src/tex/variables.dat"
    cache:
        #True
        False
    script:
        "src/scripts/bioverse_pipeline.py"