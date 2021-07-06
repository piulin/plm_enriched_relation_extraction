"""
-------------------------------------------------------------------------------------
Exploring Linguistically Enriched Transformers for Low-Resource Relation Extraction:
    --Enriched Attention on PLM

    by Pedro G. Bascoy  (Bosch Center for Artificial Intelligence (BCAI)),

    with the supervision of

    Prof. Dr. Sebastian Padó (Institut für Machinelle Sprachverarbeitung (IMS)),
    and Dr. Heike Adel-Vu  (BCAI).
-------------------------------------------------------------------------------------
"""


"""
Main function: Parsers the command-line arguments and passes the control to the main engine.
"""

import engine
from args import argprs


def main():

    # Get the command-line parser
    prs = argprs.parser()

    # Parse command-line arguments
    args = prs.parse_args()

    print(args)

    # Pass the control to the engine module
    engine.run(args)

if __name__ == '__main__':

    main()
