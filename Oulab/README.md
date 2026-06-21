# Local OULAD Data

The OULAD files are required to run the experiments but are deliberately not
included in this repository. Download the data from an authorized source, such
as the dataset publication at <https://doi.org/10.1038/sdata.2017.171>, and
comply with the dataset's applicable terms.

Place the following files directly in this directory:

```text
assessments.csv
courses.csv
studentAssessment.csv
studentInfo.csv
studentRegistration.csv
studentVle.csv
vle.csv
```

The default configurations use `data_dir: Oulab`. To use another directory,
copy a configuration file from `config/` and change its `data_dir` value.

Do not commit the CSV files or any learner-level derived outputs to a public
fork. This directory is configured to ignore all files except this README.
