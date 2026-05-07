#!/usr/bin/env python3
"""
ATSE Visualization Script
Separate script for visualizing ATSE events using leafletfa_utils
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import List, Set

from atsemapper.atseviz.main import *

def extract_unique_transcripts(juncs: pd.DataFrame) -> List[str]:
    """
    Extract unique transcripts from all transcript-related columns in junction DataFrame.
    """
    transcript_columns = ["perfect_match_5_prime", "perfect_match_3_prime"]
    all_transcripts: Set[str] = set()
    
    for column in transcript_columns:
        if column not in juncs.columns:
            continue
            
        for value in juncs[column]:
            if pd.isna(value) or value == 'NA' or value is None:
                continue
                
            if isinstance(value, str):
                if value.strip() == '':
                    continue
                for transcript in value.split(','):
                    transcript = transcript.strip()
                    if transcript and transcript != 'NA':
                        all_transcripts.add(transcript)
            elif isinstance(value, list):
                for transcript in value:
                    if transcript and not pd.isna(transcript) and transcript != 'NA':
                        all_transcripts.add(str(transcript))
    print(f"Unique transcripts: {all_transcripts}")
    return sorted(list(all_transcripts))

def visualize_atse_event(atse_event, atse_df, db, species="human", 
                        output_dir=None, padding=5000, base_width=10, 
                        trans_height=1, show_usage=False, show_junc_lines=False,
                        filter_ensembl_transcripts=True):
    """
    Process and visualize a single ATSE event.
    
    Args:
        atse_event (str): The ID of the ATSE event to process
        atse_df (pd.DataFrame): DataFrame containing ATSE data
        db: Database object for genomic annotations
        species (str): "human" or "mouse"
        output_dir (str): Directory to save figures
        padding (int): Number of base pairs to add on each side
        base_width (int): Width parameter for plotting
        trans_height (int): Height parameter for transcript display
        show_usage (bool): Whether to show junction usage information
        show_junc_lines (bool): Whether to show junction lines
        filter_ensembl_transcripts (bool): Whether to filter for ENST/ENSMUST transcripts
        
    Returns:
        dict: A dictionary containing processed data
    """
    try:
        # Filter junctions for the specific ATSE event
        juncs = atse_df[atse_df["event_id"] == atse_event].copy()
        
        if juncs.empty:
            print(f"No junctions found for ATSE event: {atse_event}")
            return None
        
        # Calculate usage ratio and add cluster information
        if "total_score" in juncs.columns:
            juncs.loc[:, "usage_ratio"] = juncs["total_score"] / juncs["total_score"].sum()
        juncs.loc[:, "Cluster"] = juncs["event_id"]
        
        # Convert junction IDs to a format suitable for plotting
        splice_junctions = convert_junction_ids(juncs)
        
        # Extract unique transcripts mentioned in the junctions
        unique_transcripts = extract_unique_transcripts(juncs)
        
        if not unique_transcripts:
            print(f"No transcripts found for ATSE event: {atse_event}")
            return None
        
        if filter_ensembl_transcripts:
            # Ensure unique transcripts start either with ENST or ENSMUST
            unique_transcripts_keep = [t for t in unique_transcripts if t and t.startswith(('ENST', 'ENSMUST'))]
            unique_transcripts_remove = [t for t in unique_transcripts if t and not t.startswith(('ENST', 'ENSMUST'))]
            print(f"Unique transcripts: {unique_transcripts_keep}")
            print(f"Unique transcripts to remove: {unique_transcripts_remove}")
        else:
            unique_transcripts_keep = unique_transcripts

        # Fetch transcript data from the database
        transcript_data = fetch_transcripts_and_annotations(db, unique_transcripts_keep)
        
        # Create mapping between transcript IDs, names, and types
        conversions = {}
        for t in transcript_data.keys():
            transcript_id = transcript_data[t]["transcript_id"]
            transcript_name = transcript_data[t]["transcript_name"]
            transcript_type = transcript_data[t]["transcript_type"]
            conversions[transcript_id] = {
                "transcript_name": transcript_name,
                "transcript_type": transcript_type,
                "transcript_id": transcript_id
            }
        
        # Convert mappings to DataFrame for easier access
        conversions_df = pd.DataFrame.from_dict(conversions, orient='index')
        
        # Determine region boundaries for plotting
        region_start, region_end = determine_region_boundaries(splice_junctions)
        
        # Set filename
        if output_dir:
            timestamp = datetime.now().strftime("%H%M%S")
            gene_name = juncs['gene_name'].iloc[0] if 'gene_name' in juncs.columns else "unknown"
            filename = os.path.join(output_dir, f"{timestamp}_{species}_{gene_name}_{atse_event}.pdf")
        else:
            filename = None
        
        # Plot with the appropriate parameters
        plot_exons_and_junctions(
            db, 
            atse_event, 
            transcript_data, 
            splice_junctions, 
            region_start - padding, 
            region_end + padding,
            base_width=base_width, 
            trans_height=trans_height, 
            show_usage=show_usage, 
            show_junc_lines=show_junc_lines, 
            filename=filename
        )
        
        print(f"ATSE visualization saved: {filename}")
        
        # Return the processed data
        return {
            "junctions": juncs,
            "splice_junctions": splice_junctions,
            "transcript_data": transcript_data,
            "conversions_df": conversions_df,
            "gene_name": juncs['gene_name'].iloc[0] if 'gene_name' in juncs.columns else "Unknown",
            "filename": filename
        }
        
    except Exception as e:
        print(f"Error processing ATSE event {atse_event}: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_examples_from_shared_genes(mouse_atse_df, human_atse_df, 
                                       db_mouse, db_human, shared_genes, 
                                       output_dir, n_examples=3):
    """
    Visualize example ATSE events from shared orthologous genes.
    """
    print(f"\nVisualizing {n_examples} example ATSE events from shared genes...")
    
    shared_gene_list = list(shared_genes)[:n_examples]
    results = []
    
    for gene_name in shared_gene_list:
        print(f"\n--- Processing gene: {gene_name} ---")
        
        # Get mouse ATSEs for this gene
        mouse_gene_atses = mouse_atse_df[mouse_atse_df['gene_name'] == gene_name]
        if not mouse_gene_atses.empty:
            mouse_atse_id = mouse_gene_atses['event_id'].iloc[0]
            print(f"Visualizing mouse ATSE: {mouse_atse_id}")
            mouse_result = visualize_atse_event(
                mouse_atse_id, mouse_atse_df, db_mouse, 
                species="mouse", output_dir=output_dir
            )
            if mouse_result:
                results.append(("mouse", gene_name, mouse_result))
        
        # Get human ATSEs for this gene
        # Note: human data might use 'Mouse gene name' column for the ortholog mapping
        human_gene_atses = human_atse_df[
            (human_atse_df.get('gene_name', '') == gene_name) | 
            (human_atse_df.get('Mouse gene name', '') == gene_name)
        ]
        
        if not human_gene_atses.empty:
            human_atse_id = human_gene_atses['event_id'].iloc[0]
            print(f"Visualizing human ATSE: {human_atse_id}")
            human_result = visualize_atse_event(
                human_atse_id, human_gene_atses, db_human, 
                species="human", output_dir=output_dir
            )
            if human_result:
                results.append(("human", gene_name, human_result))
    
    print(f"\nCompleted visualization of {len(results)} ATSE events")
    return results

