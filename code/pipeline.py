#Version: v1.0
#Date Last Updated: 04-12-2026

#%% MODULE BEGINS
module_name = "pipeline"

"""
Version: v1.0

Description:
    Sequential pipeline runner for the CMPS 470 stellar classification project.
    Executes all processing stages in order so the full project can be run
    from a single entry point.

Authors:
    Group B

Date Created     : 04-12-2026
Date Last Updated: 04-29-2026

Doc:
    Stages (in order):
        1. preprocess     — data cleaning, splitting, z-score scaling  (PA1)
        2. feature_extract — correlation analysis and PCA               (PA2)

    Future PA modules should be imported and appended to main() here.

Notes:
    All stage modules must live in the same directory as this file (code/).
    Each stage module must expose a main() function.
"""

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
from pathlib import Path

if __name__ == "__main__":
    # Ensure sibling modules are importable regardless of working directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
#

#custom imports
import classifier
import feature_extract
import preprocess

#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PIPELINE_STAGES = ("preprocess", "feature_extract", "classifier")

#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here


#Class definitions Start Here


#Function definitions Start Here
def main():
    print(f"Pipeline begins. Stages: {PIPELINE_STAGES}\n")

    preprocess.main()
    print()

    feature_extract.main()
    print()

    classifier.main()
    print()

    print("Pipeline completed successfully.")
#


#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here



#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":

    print(f'"{module_name}" module begins.')

    #TEST Code
    main()
