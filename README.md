# medical-text-simplification
NLP to simplify medication text while preserving meaning
# Medical Text Simplification Project
## Overview
This project focuses on simplifying medication instructions using GPT-4o while preserving meaning and critical numeric safety information. Medical instruction labels are written above reading literacy in the U.S. The goal of this project is to reduce reading complexity while maintaining clinical accuracy.

## Dataset
Medication text was collected from FDA DailyMed.

## Methodology
- GPT-4o used for text simplification
- The prompt enforced:
  -   Target reading level (6-8th grade)
  -   Simple, clear language
-   No change in numeric values (dosage & time) 
- Meaning preservation was evaluated using SentenceTransformer (all-mpnet-base-v2) with cosine similarity.
- Readability was measured using: 
  -   Flesch-Kincaid for readability/ease 
  -   Numeric token checks for safety
- Numeric token checks implemented to detect any loss of important medical values.

## Results
- Reduced reading grade levels across all medications
- Increased reading ease scores
- High meaning preservation for most samples (generally ~.7-.9).
- Minimal numeric loss, showing safety information was preserved

## Key Findings
- Simplification effectiveness varied based on the complexity of the original text.
- The more complex and dense text needed more aggressive simplification to meet the grade level threshold, which slightly reduced meaning.
- The more structured text handled the simplification better and allowed for better meaning preservation
- Example:
  - Amlodipine required aggressive simplification leading to a lower meaning score than the rest
  - Albuterol showed more balanced simplification and maintained high meaning preservation. 

## Limitations
- The dataset is small (20 medication, 40 entries( dosages & warning)), which may limit generalizability to real clinical settings where medication instructions can vary in structure and complexity. 
- Results can vary depending on the structure of the original medical text.

## Future Work
- Expand dataset to include a more diverse range of medications.
- Validate the approach across different types of medical text. 
- Improve consistency across different types of text structures.

## Project Structure
- 01_build_and_run.ipynb -> main notebood for running pipeline
- src/
  -  run_pipeline.py -> runs full simplification pipeline
  -  utils.py -> helper functions (text cleaning, numeric checks, simplification logic)
- drug_summary.csv -> dataset containing original medication text
- medication_dataset_processed.csv -> processed data used for analysis
- final_pipeline_ready.csv -> cleaned dataset ready for results
- final_results_full.csv -> full output results
- final_results_table.csv -> final summarized results table

## How to Run
- Open notebook in VS code
- Run cells in order

## Presentation


