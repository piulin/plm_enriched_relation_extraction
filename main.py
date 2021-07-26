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

from args import argprs
from typing import Dict, Union


def main() -> None:
    """
    Parses the command-line arguments and hands over the control to the engine
    :return:
    """

    # Get the command-line parser
    prs : argprs.parser = argprs.parser()

    # Parse command-line arguments
    args : dict = prs.parse_args()

    print(f"Command-line args: {args}")

    # Pass the control to the engine module
    import engine
    engine.run(args)

if __name__ == '__main__':

    main()
