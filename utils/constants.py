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
This module is intended to keep the constants and literals
"""

# Entity Marker Tokens (EMT)
E1S: str = '<e1>'
E1E: str = '</e1>'
E2S: str = '<e2>'
E2E: str = '</e2>'

# padding index used for sdp flag
sdp_flag_padding_index: int = 0
